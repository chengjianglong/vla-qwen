"""Download LIBERO (LeRobot format, physical-intelligence/libero) and convert it into
fast random-access numpy arrays:

  data/libero/images.u8      uint8 memmap [N, 2, R, R, 3]  (agentview, wrist)
  data/libero/arrays.npz     state [N,8], actions [N,7], episode/frame/task index [N]
  data/libero/meta.json      tasks, episode boundaries, normalization stats

Usage:  python prepare_data.py --res 224 --workers 16
"""
import argparse, io, json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download

REPO = "physical-intelligence/libero"


def decode(cell, res):
    b = cell["bytes"] if isinstance(cell, dict) else cell
    im = Image.open(io.BytesIO(b)).convert("RGB")
    if im.size != (res, res):
        im = im.resize((res, res), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def process_episode(args):
    path, start, total, res, out_path = args
    t = pq.read_table(path).to_pydict()
    n = len(t["frame_index"])
    mm = np.memmap(out_path, dtype=np.uint8, mode="r+", shape=(total, 2, res, res, 3))
    for i in range(n):
        mm[start + i, 0] = decode(t["image"][i], res)
        mm[start + i, 1] = decode(t["wrist_image"][i], res)
    mm.flush()
    del mm
    return (start, np.asarray(t["state"], np.float32), np.asarray(t["actions"], np.float32),
            np.asarray(t["episode_index"], np.int64), np.asarray(t["frame_index"], np.int64),
            np.asarray(t["task_index"], np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/libero_raw")
    ap.add_argument("--out", default="data/libero")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    raw = snapshot_download(REPO, repo_type="dataset", local_dir=a.raw, max_workers=8)
    os.makedirs(a.out, exist_ok=True)

    episodes = [json.loads(l) for l in open(os.path.join(raw, "meta/episodes.jsonl"))]
    episodes.sort(key=lambda e: e["episode_index"])
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(os.path.join(raw, "meta/tasks.jsonl"))}
    starts, s = [], 0
    for e in episodes:
        starts.append(s)
        s += e["length"]
    total = s
    print(f"{len(episodes)} episodes, {total} frames, {len(tasks)} tasks")

    img_path = os.path.join(a.out, "images.u8")
    np.memmap(img_path, dtype=np.uint8, mode="w+", shape=(total, 2, a.res, a.res, 3)).flush()

    jobs = []
    for e, st in zip(episodes, starts):
        ei = e["episode_index"]
        p = os.path.join(raw, f"data/chunk-{ei // 1000:03d}/episode_{ei:06d}.parquet")
        jobs.append((p, st, total, a.res, img_path))

    state = np.zeros((total, 8), np.float32)
    actions = np.zeros((total, 7), np.float32)
    ep = np.zeros(total, np.int64); fr = np.zeros(total, np.int64); tk = np.zeros(total, np.int64)
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (st, sa, ac, e_, f_, t_) in enumerate(ex.map(process_episode, jobs, chunksize=4)):
            n = len(f_)
            state[st:st + n], actions[st:st + n] = sa, ac
            ep[st:st + n], fr[st:st + n], tk[st:st + n] = e_, f_, t_
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} episodes", flush=True)

    np.savez(os.path.join(a.out, "arrays.npz"), state=state, actions=actions,
             episode_index=ep, frame_index=fr, task_index=tk)

    def stats(x):
        return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                "q01": np.quantile(x, 0.01, 0).tolist(), "q99": np.quantile(x, 0.99, 0).tolist()}

    meta = {
        "res": a.res, "total": total,
        "tasks": {int(k): v for k, v in tasks.items()},
        "episode_starts": starts, "episode_lengths": [e["length"] for e in episodes],
        "stats": {"state": stats(state), "actions": stats(actions)},
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
