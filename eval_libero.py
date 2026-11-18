"""Closed-loop evaluation of a trained checkpoint in the LIBERO simulator.

Observation processing mirrors the training data (physical-intelligence/libero):
  images  : agentview + wrist, 256x256 render rotated 180 deg, resized to the training resolution (224)
  state   : eef_pos (3) + axis-angle(eef_quat) (3) + gripper_qpos (2)
  actions : 7-D delta EEF + gripper (-1 open / +1 close); the predicted chunk is executed open-loop
            for --replan steps, then the model is queried again.

Run with the LIBERO env (see README):
  ./run_eval.sh --ckpt runs/qwen3vl2b_libero_2k/ckpt_002000.pt --trials 10
"""
import argparse, json, math, os, time

import imageio
import numpy as np
import torch
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla import Normalizer, QwenVLA

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_WAIT = 10  # let objects settle after reset (standard LIBERO eval protocol)
DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(q):
    """robosuite (x, y, z, w) quaternion -> axis-angle (same as the dataset conversion)."""
    q = np.array(q, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(q[3]) / den


def prep_image(img, res, crop):
    img = np.ascontiguousarray(img[::-1, ::-1])  # LIBERO renders upside down; dataset is rotated 180 deg
    im = Image.fromarray(img).resize((res, res), Image.BICUBIC)
    if crop < 1.0:  # center crop matching the average random-resized-crop used in training
        s = int(round(res * math.sqrt(crop)))
        o = (res - s) // 2
        im = im.crop((o, o, o + s, o + s)).resize((res, res), Image.BILINEAR)
    return np.asarray(im)


def get_state(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


class Policy:
    def __init__(self, ckpt, dev):
        run = os.path.dirname(ckpt)
        args = json.load(open(os.path.join(run, "args.json")))
        stats = json.load(open(os.path.join(run, "norm_stats.json")))
        self.res = json.load(open(os.path.join(args["data"], "meta.json")))["res"] if os.path.exists(
            os.path.join(args["data"], "meta.json")) else 224
        self.chunk = args["chunk"]
        self.model = QwenVLA(args["base"], chunk=self.chunk, lora_r=args["lora_r"], grad_ckpt=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        loaded = set(sd)
        assert all(k in loaded for k in self.model.trainable_state_dict()), "checkpoint is missing trainable weights"
        self.model.to(dev).eval()
        self.dev = dev
        self.snorm = Normalizer(stats["state"])
        self.anorm = Normalizer(stats["actions"], skip_last=True)

    @torch.no_grad()
    def __call__(self, scene, wrist, state, task):
        enc = self.model.encode([[scene, wrist]], [task]).to(self.dev)
        st = torch.from_numpy(self.snorm.norm(state)[None].astype(np.float32)).to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(enc, st)[0].float().cpu().numpy()  # [chunk, 7] normalized
        act = self.anorm.unnorm(pred)
        act[:, -1] = np.where(pred[:, -1] > 0, 1.0, -1.0)  # binarize gripper
        return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suites", default="libero_spatial,libero_object,libero_goal,libero_10")
    ap.add_argument("--trials", type=int, default=10, help="episodes per task (official protocol: 50)")
    ap.add_argument("--tasks", default="", help="comma-separated task ids to run (default: all)")
    ap.add_argument("--replan", type=int, default=8, help="actions executed per model query")
    ap.add_argument("--crop", type=float, default=0.95, help="center-crop area fraction (1 = none)")
    ap.add_argument("--videos", type=int, default=2, help="save this many videos per task")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    out = a.out or os.path.join(os.path.dirname(a.ckpt), "eval_" + os.path.basename(a.ckpt).split(".")[0])
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda", 0)
    pol = Policy(a.ckpt, dev)
    replan = min(a.replan, pol.chunk)

    res_path = os.path.join(out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    bench = benchmark.get_benchmark_dict()
    for suite_name in a.suites.split(","):
        suite = bench[suite_name]()
        task_ids = [int(t) for t in a.tasks.split(",")] if a.tasks else range(suite.n_tasks)
        for tid in task_ids:
            key = f"{suite_name}/{tid}"
            if key in results and results[key]["trials"] >= a.trials:
                continue
            task = suite.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            init_states = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder,
                                                  task.init_states_file), weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(a.seed)
            succ, lens, t0 = 0, [], time.time()
            for ep in range(a.trials):
                env.reset()
                obs = env.set_init_state(init_states[ep % len(init_states)])
                frames, plan, done, t = [], [], False, 0
                for t in range(MAX_STEPS[suite_name] + NUM_WAIT):
                    if t < NUM_WAIT:
                        obs, _, done, _ = env.step(DUMMY)
                        continue
                    scene = prep_image(obs["agentview_image"], pol.res, a.crop)
                    wrist = prep_image(obs["robot0_eye_in_hand_image"], pol.res, a.crop)
                    if ep < a.videos:
                        frames.append(np.concatenate([obs["agentview_image"][::-1, ::-1],
                                                      obs["robot0_eye_in_hand_image"][::-1, ::-1]], 1))
                    if not plan:
                        plan = list(pol(scene, wrist, get_state(obs), task.language)[:replan])
                    obs, _, done, _ = env.step(plan.pop(0).tolist())
                    if done:
                        break
                succ += int(done); lens.append(t - NUM_WAIT + 1)
                if frames:
                    tag = "success" if done else "fail"
                    imageio.mimwrite(os.path.join(out, "videos", f"{suite_name}_t{tid:02d}_ep{ep}_{tag}.mp4"),
                                     frames, fps=20, quality=7, macro_block_size=1)
                print(f"{key} ep {ep}: {'SUCCESS' if done else 'fail'} ({lens[-1]} steps)  "
                      f"running {succ}/{ep + 1}", flush=True)
            env.close()
            results[key] = {"task": task.language, "success": succ, "trials": a.trials,
                            "mean_len": float(np.mean(lens)), "seconds": time.time() - t0}
            json.dump(results, open(res_path, "w"), indent=1)
            print(f"== {key} {task.language!r}: {succ}/{a.trials}", flush=True)

    print("\n==== summary ====")
    for suite_name in a.suites.split(","):
        r = [v for k, v in results.items() if k.startswith(suite_name + "/")]
        if r:
            s, n = sum(v["success"] for v in r), sum(v["trials"] for v in r)
            print(f"{suite_name:15s} {s:4d}/{n:<4d} = {100 * s / n:5.1f}%  ({len(r)} tasks)")
    print("results ->", res_path)


if __name__ == "__main__":
    main()
