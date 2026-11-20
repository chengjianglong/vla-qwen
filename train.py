"""Train the Qwen3-VL VLA on LIBERO.

Single GPU:  python train.py
Both GPUs:   torchrun --nproc_per_node 2 train.py
Smoke test:  python train.py --steps 20 --log_every 1 --save_every 0
"""
import argparse, json, math, os, random, time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from vla import Normalizer, QwenVLA


class LiberoDataset(Dataset):
    def __init__(self, root, chunk):
        self.meta = json.load(open(os.path.join(root, "meta.json")))
        a = np.load(os.path.join(root, "arrays.npz"))
        self.state, self.actions = a["state"], a["actions"]
        self.task_index, self.episode_index = a["task_index"], a["episode_index"]
        R, N = self.meta["res"], self.meta["total"]
        self.img_path, self.shape = os.path.join(root, "images.u8"), (N, 2, R, R, 3)
        self.images = None  # opened lazily per worker
        self.chunk = chunk
        # last frame index (global) of each frame's episode, for chunk padding
        ends = np.array(self.meta["episode_starts"]) + np.array(self.meta["episode_lengths"]) - 1
        self.ep_end = ends[np.searchsorted(self.meta["episode_starts"], np.arange(N), side="right") - 1]
        self.snorm = Normalizer(self.meta["stats"]["state"])
        self.anorm = Normalizer(self.meta["stats"]["actions"], skip_last=True)
        self.tasks = {int(k): v for k, v in self.meta["tasks"].items()}

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.memmap(self.img_path, dtype=np.uint8, mode="r", shape=self.shape)
        imgs = np.array(self.images[i])  # [2, R, R, 3]
        idx = np.minimum(np.arange(i, i + self.chunk), self.ep_end[i])
        return {
            "images": imgs,
            "task": self.tasks[int(self.task_index[i])],
            "state": self.snorm.norm(self.state[i]).astype(np.float32),
            "actions": self.anorm.norm(self.actions[idx]).astype(np.float32),
        }


def augment(img, rng):
    """Light random-resized-crop (90% area) + brightness/contrast jitter, uint8 HxWx3 -> uint8."""
    R = img.shape[0]
    s = int(R * math.sqrt(rng.uniform(0.9, 1.0)))
    y, x = rng.integers(0, R - s + 1), rng.integers(0, R - s + 1)
    t = torch.from_numpy(img[y:y + s, x:x + s]).permute(2, 0, 1)[None].float()
    t = torch.nn.functional.interpolate(t, size=(R, R), mode="bilinear", align_corners=False)[0]
    t = (t - t.mean()) * rng.uniform(0.8, 1.2) + t.mean() * rng.uniform(0.8, 1.2)
    return t.clamp(0, 255).byte().permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/libero")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", default="runs/qwen3vl2b_libero")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4, help="per-GPU micro batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=50000, help="optimizer steps")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--resume", default="")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    rng = np.random.default_rng(rank)

    torch.manual_seed(0)  # identical init on every rank, so DDP needs no initial broadcast
    model = QwenVLA(a.base, chunk=a.chunk, lora_r=a.lora_r).to(dev)
    torch.manual_seed(rank)
    step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        step0 = ck["step"]
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    if a.resume and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    for _ in range(step0):
        sched.step()

    # DDP over just the trainable module tree; frozen params are ignored
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False,
                                                       init_sync=False) if ddp else model

    ds = LiberoDataset(a.data, a.chunk)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=step0) if ddp else None  # new order on resume
    dl = DataLoader(ds, batch_size=a.bs, sampler=sampler, shuffle=sampler is None, num_workers=a.workers,
                    collate_fn=lambda b: b, drop_last=True, persistent_workers=True, prefetch_factor=4)

    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
        json.dump(ds.meta["stats"], open(os.path.join(a.out, "norm_stats.json"), "w"))
        tb = SummaryWriter(a.out)

    step, micro, epoch, t0, loss_acc = step0, 0, 0, time.time(), 0.0
    model.train()
    while step < a.steps:
        if sampler:
            sampler.set_epoch(epoch)
        for batch in dl:
            imgs = [[augment(im, rng) if not a.no_aug else im for im in b["images"]] for b in batch]
            enc = model.encode(imgs, [b["task"] for b in batch]).to(dev)
            state = torch.from_numpy(np.stack([b["state"] for b in batch])).to(dev)
            target = torch.from_numpy(np.stack([b["actions"] for b in batch])).to(dev)

            sync = (micro + 1) % a.accum == 0
            ctx = dmodel.no_sync() if (ddp and not sync) else torch.enable_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dmodel(enc, state)
                loss = torch.nn.functional.l1_loss(pred, target)
                (loss / a.accum).backward()
            loss_acc += loss.item() / a.accum
            micro += 1
            if not sync:
                continue

            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % a.log_every == 0:
                dt = (time.time() - t0) / a.log_every
                mem = torch.cuda.max_memory_allocated(dev) / 2**30
                print(f"step {step} loss {loss_acc:.4f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{dt:.2f}s/it mem {mem:.2f}G", flush=True)
                tb.add_scalar("loss", loss_acc, step); tb.add_scalar("lr", sched.get_last_lr()[0], step)
                t0 = time.time()
            loss_acc = 0.0
            if rank == 0 and a.save_every and (step % a.save_every == 0 or step == a.steps):
                torch.save({"model": model.trainable_state_dict(), "opt": opt.state_dict(), "step": step},
                           os.path.join(a.out, f"ckpt_{step:06d}.pt"))
            if step >= a.steps:
                break
        epoch += 1

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
