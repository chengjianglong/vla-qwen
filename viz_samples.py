"""Visualize a few LIBERO training samples exactly as the dataset serves them.

Each row = one sample (random frame from an episode of a different task):
  scene cam | wrist cam | translation actions over the episode | rotation actions over the episode
The shaded band marks the 8-step action chunk the model is trained to predict from that frame;
gray background = gripper closed.

Usage:  python viz_samples.py --n 4 --out samples.png
"""
import argparse, json, os, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (x/y/z, also roll/pitch/yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/libero")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="samples.png")
    a = ap.parse_args()

    m = json.load(open(os.path.join(a.data, "meta.json")))
    arr = np.load(os.path.join(a.data, "arrays.npz"))
    R, N = m["res"], m["total"]
    imgs = np.memmap(os.path.join(a.data, "images.u8"), np.uint8, "r", shape=(N, 2, R, R, 3))
    starts, lens = np.array(m["episode_starts"]), np.array(m["episode_lengths"])
    ep_task = arr["task_index"][starts]

    rng = np.random.default_rng(a.seed)
    tasks = rng.choice(np.unique(ep_task), a.n, replace=False)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig = plt.figure(figsize=(15, 3.1 * a.n), facecolor=SURF)
    gs = fig.add_gridspec(a.n, 4, width_ratios=[1, 1, 1.7, 1.7], hspace=0.55, wspace=0.25)

    for r, t in enumerate(tasks):
        e = rng.choice(np.flatnonzero(ep_task == t))
        s, L = starts[e], lens[e]
        f = int(rng.integers(0, L - a.chunk))
        g = s + f
        act = arr["actions"][s:s + L]
        steps = np.arange(L)
        closed = act[:, 6] > 0

        for c, (name, im) in enumerate([("scene camera", imgs[g, 0]), ("wrist camera", imgs[g, 1])]):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, color=INK2, fontsize=8, loc="left")
        fig.text(0.125, ax.get_position().y1 + 0.028 / a.n * 4,
                 f"Task {t}: “{textwrap.shorten(m['tasks'][str(t)], 110)}”   "
                 f"(episode {e}, frame {f}/{L})", fontsize=10, weight="bold")

        for c, (lo, labels, title) in enumerate([(0, ["x", "y", "z"], "translation Δ (raw)"),
                                                 (3, ["roll", "pitch", "yaw"], "rotation Δ (raw)")]):
            ax = fig.add_subplot(gs[r, 2 + c]); ax.set_facecolor(SURF)
            ylo, yhi = act[:, lo:lo + 3].min() * 1.1, act[:, lo:lo + 3].max() * 1.1
            ax.fill_between(steps, ylo, yhi, where=closed, color="#efeeea", step="mid", lw=0)
            ax.axvspan(f, f + a.chunk, color="#2a78d6", alpha=0.15, lw=0)
            for k in range(3):
                ax.plot(steps, act[:, lo + k], color=C[k], lw=1.6, label=labels[k])
            ax.set_ylim(ylo, yhi); ax.set_xlim(0, L - 1)
            ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(title, color=INK2, fontsize=8, loc="left")
            ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=7.5, handlelength=1.2)
            if r == a.n - 1:
                ax.set_xlabel("timestep in episode")

    fig.text(0.125, 0.015, "Blue band = 8-step action chunk the model predicts from the shown frame.  "
             "Gray background = gripper closed.", color=INK2, fontsize=8.5)
    fig.savefig(a.out, dpi=110, facecolor=SURF, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
