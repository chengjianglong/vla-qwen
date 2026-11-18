"""Generate the README figures in media/:

  architecture.svg   model diagram (with real camera frames embedded)
  model_io.gif       one demo: scene + wrist camera, with the 7-D action trace and the 8-step chunk window
  demos.gif          2x2 grid of demonstrations from four different LIBERO task suites
  train_loss.png     training loss curve parsed from a train.py log
  rollouts.gif       closed-loop LIBERO rollouts of the trained policy (from eval_libero.py videos)

Usage:  python make_media.py [--log runs_2k.log]
"""
import argparse, base64, io, json, os, re, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                      "#e87ba4", "#008300", "#6250d6")
OUT = "media"


def font(size, bold=False):
    for p in ([f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"]):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


class Data:
    def __init__(self, root):
        self.m = json.load(open(os.path.join(root, "meta.json")))
        self.a = np.load(os.path.join(root, "arrays.npz"))
        R, N = self.m["res"], self.m["total"]
        self.img = np.memmap(os.path.join(root, "images.u8"), np.uint8, "r", shape=(N, 2, R, R, 3))
        self.starts = np.array(self.m["episode_starts"]); self.lens = np.array(self.m["episode_lengths"])
        self.ep_task = self.a["task_index"][self.starts]

    def episode(self, task, k=0):
        e = np.flatnonzero(self.ep_task == task)[k]
        s, L = self.starts[e], self.lens[e]
        return e, self.img[s:s + L], self.a["actions"][s:s + L], self.m["tasks"][str(task)]


# ----------------------------------------------------------------------------------------------
def architecture_svg(d, path):
    _, ims, _, task = d.episode(10)
    f = len(ims) // 3

    def b64(im):
        buf = io.BytesIO(); Image.fromarray(np.asarray(im)).resize((112, 112)).save(buf, "JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    W, H = 980, 630
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
         f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" rx="14" fill="{SURF}"/>',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']

    def box(x, y, w, h, fill, stroke, label, sub=None, tc=INK, r=10, fs=14):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        ty = y + h / 2 + (fs * 0.35 if sub is None else -2)
        s.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="{fs}" font-weight="600" fill="{tc}">{label}</text>')
        if sub:
            s.append(f'<text x="{x + w / 2}" y="{ty + 17}" text-anchor="middle" font-size="11.5" fill="{INK2}">{sub}</text>')

    def arrow(x1, y1, x2, y2):
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.6" marker-end="url(#ar)"/>')

    def label(x, y, t, fs=12, c=INK2, anchor="middle", w="400"):
        s.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{w}" fill="{c}">{t}</text>')

    # --- inputs
    for i, (nm, im) in enumerate([("scene camera", ims[f, 0]), ("wrist camera", ims[f, 1])]):
        x = 40 + i * 128
        s.append(f'<image x="{x}" y="36" width="112" height="112" xlink:href="data:image/jpeg;base64,{b64(im)}"/>')
        s.append(f'<rect x="{x}" y="36" width="112" height="112" fill="none" stroke="{GRID}"/>')
        label(x + 56, 166, nm, 11.5)
    box(330, 62, 300, 60, "#f0efec", GRID, f'"{task}"', "language instruction", fs=13)
    box(660, 62, 130, 60, "#fdf0ea", ORANGE, "robot state", "8-D (eef pose, gripper)", fs=13)
    box(820, 62, 130, 60, "#e8f7f1", AQUA, "8 action queries", "learned embeddings", fs=12.5)
    label(40, 22, "INPUTS", 11, MUTED, "start", "700")

    # --- encoders
    box(40, 196, 240, 50, "#eaf2fc", BLUE, "Qwen3-VL vision encoder", "frozen · 2×49 visual tokens", fs=13)
    box(330, 196, 300, 50, "#f0efec", GRID, "tokenizer + embedding", "frozen (kept on CPU)", fs=13)
    box(660, 196, 130, 50, "#fdf0ea", ORANGE, "state MLP", "8 → 2048", fs=13)
    arrow(160, 176, 160, 194); arrow(480, 124, 480, 194); arrow(725, 124, 725, 194)

    # --- token sequence (group centers drive arrows + labels)
    y, x, tw, step = 272, 40, 12, 15
    seq = [("scene", BLUE, 14), ("sep", MUTED, 2), ("wrist", BLUE, 14), ("text", MUTED, 12),
           ("state", ORANGE, 1), ("sep2", MUTED, 2), ("act", AQUA, 8)]
    cen, xs_act = {}, []
    for name, col, n in seq:
        x_start = x
        for _ in range(n):
            s.append(f'<rect x="{x}" y="{y}" width="{tw}" height="22" rx="3" fill="{col}"/>')
            if name == "act":
                xs_act.append(x + tw / 2)
            x += step
        cen[name] = (x_start + x - step + tw) / 2
    arrow(110, 248, cen["scene"], 270); arrow(210, 248, cen["wrist"], 270)
    arrow(480, 248, cen["text"], 270); arrow(725, 248, cen["state"], 270)
    s.append(f'<path d="M885,124 V254 H{cen["act"]} V268" stroke="{MUTED}" stroke-width="1.6" fill="none" marker-end="url(#ar)"/>')
    label(cen["scene"], 312, "scene tokens", 11, BLUE)
    label(cen["wrist"], 312, "wrist tokens", 11, BLUE)
    label(cen["text"], 312, "task text", 11, MUTED)
    label(cen["state"], 326, "state", 11, ORANGE)
    label(x + 4, 287, "action", 11, AQUA, "start"); label(x + 4, 300, "queries", 11, AQUA, "start")

    # --- LLM
    box(40, 336, 910, 60, "#f1effc", VIOLET, "Qwen3-VL-2B language model  (28 layers, frozen + LoRA r=32 on all linear layers)",
        "one forward pass · each action query attends to both images, the instruction and the state", fs=14)
    for xx in xs_act:
        arrow(xx, 296, xx, 334)

    # --- head + output
    for xx in xs_act:
        arrow(xx, 398, xx, 428)
    mid = cen["act"]
    hx0, hx1 = mid - 125, mid + 125
    box(hx0, 430, hx1 - hx0, 50, "#e8f7f1", AQUA, "action head (shared)", "LayerNorm → MLP 2048→1024→7", fs=13)
    arrow(mid, 482, mid, 506)
    # output chunk heat strip: rows = 8 future steps, cols = 7 action dims
    _, _, act, _ = d.episode(10)
    fo = 44
    lo = np.array(d.m["stats"]["actions"]["q01"]); hi = np.array(d.m["stats"]["actions"]["q99"])
    chunk = np.clip(2 * (act[fo:fo + 8] - lo) / (hi - lo) - 1, -1, 1)
    cw, ch = 22, 9
    x0 = mid - 4 * cw
    for t in range(8):
        for k in range(7):
            v = chunk[t, k]
            c = np.array([0x2a, 0x78, 0xd6]) if v < 0 else np.array([0xeb, 0x68, 0x34])
            c = (np.array([0xf0, 0xef, 0xec]) * (1 - abs(v)) + c * abs(v)).astype(int)
            s.append(f'<rect x="{x0 + t * cw + 1}" y="{508 + k * ch}" width="{cw - 2}" height="{ch - 1}" fill="#{c[0]:02x}{c[1]:02x}{c[2]:02x}"/>')
    label(x0 - 8, 524, "Δxyz", 10.5, INK2, "end"); label(x0 - 8, 551, "Δrpy", 10.5, INK2, "end")
    label(x0 - 8, 569, "grip", 10.5, INK2, "end")
    label(x0 + 8 * cw + 8, 530, "t+1 … t+8", 10.5, INK2, "start")
    label(mid, 594, "8 × 7 action chunk (normalized to [-1, 1])  →  L1 loss vs. demonstration", 12, INK, "middle", "600")
    label(mid, 612, "at test time: execute the chunk, observe, predict again", 11, INK2)

    # --- left side notes
    label(40, 448, "Trainable (41M params):", 12.5, INK, "start", "700")
    for i, t in enumerate(["LoRA adapters in the LLM", "action-query embeddings", "state MLP", "action head"]):
        label(52, 470 + i * 18, f"• {t}", 12, INK2, "start")
    label(40, 566, "Everything else (≈2B params) is frozen.", 12, INK2, "start")

    s.append("</svg>")
    open(path, "w").write("\n".join(s))


# ----------------------------------------------------------------------------------------------
def model_io_gif(d, path, task=10, stride=2, chunk=8):
    _, ims, act, text = d.episode(task)
    L = len(act)
    R = 224
    fig, axs = plt.subplots(2, 1, figsize=(4.48, 2.24 * 2), dpi=100, sharex=True, facecolor=SURF)
    for ax, (lo, labels, title) in zip(axs, [(0, "xyz", "translation Δ"), (3, ["roll", "pitch", "yaw"], "rotation Δ")]):
        ax.set_facecolor(SURF)
        ylo, yhi = act[:, lo:lo + 3].min() * 1.15, act[:, lo:lo + 3].max() * 1.15
        ax.fill_between(np.arange(L), ylo, yhi, where=act[:, 6] > 0, color="#efeeea", step="mid", lw=0)
        for k, c in enumerate([BLUE, ORANGE, AQUA]):
            ax.plot(act[:, lo + k], color=c, lw=1.5, label=labels[k])
        ax.set_ylim(ylo, yhi); ax.set_xlim(0, L - 1)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7)
        ax.set_title(title, fontsize=8, color=INK2, loc="left")
        ax.legend(loc="upper right", ncol=3, fontsize=6.5, frameon=False, handlelength=1)
    axs[1].set_xlabel("timestep", fontsize=7.5, color=INK2)
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    fh = bg.shape[0]
    to_px = lambda ax, t: ax.transData.transform((t, 0))[0]
    boxes = [(ax.get_window_extent().y0, ax.get_window_extent().y1) for ax in axs]
    plt.close(fig)

    cap_h = 46
    Wt, Ht = R + bg.shape[1], max(2 * R, fh) + cap_h
    f_cap, f_small = font(14, True), font(11)
    frames = []
    for t in range(0, L, stride):
        fr = Image.new("RGB", (Wt, Ht), SURF)
        dr = ImageDraw.Draw(fr)
        dr.text((10, 8), f'"{text}"', fill=INK, font=f_cap)
        dr.text((10, 28), "model input: both cameras + state + instruction   →   output: next 8 actions (blue band)",
                fill=INK2, font=f_small)
        fr.paste(Image.fromarray(np.asarray(ims[t, 0])), (0, cap_h))
        fr.paste(Image.fromarray(np.asarray(ims[t, 1])), (0, cap_h + R))
        plot = Image.fromarray(bg)
        pd = ImageDraw.Draw(plot, "RGBA")
        for ax_i, (y0, y1) in enumerate(boxes):
            ax = axs[ax_i]
            xa, xb = to_px(ax, t), to_px(ax, min(t + chunk, L - 1))
            pd.rectangle([xa, fh - y1, xb, fh - y0], fill=(42, 120, 214, 50))
            pd.line([xa, fh - y1, xa, fh - y0], fill=(11, 11, 11, 200), width=1)
        fr.paste(plot, (R, cap_h))
        grip = "closed" if act[t, 6] > 0 else "open"
        dr.text((6, cap_h + 6), "scene", fill="white", font=f_small, stroke_width=2, stroke_fill="black")
        dr.text((6, cap_h + R + 6), "wrist", fill="white", font=f_small, stroke_width=2, stroke_fill="black")
        dr.text((6, cap_h + 2 * R - 18), f"t = {t:3d}   gripper {grip}", fill="white", font=f_small,
                stroke_width=2, stroke_fill="black")
        frames.append(fr.quantize(128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)


# ----------------------------------------------------------------------------------------------
def demos_gif(d, path, tasks=(3, 10, 24, 36), stride=3, tile=224):
    eps = [d.episode(t) for t in tasks]
    T = max(len(a) for _, _, a, _ in eps)
    cap = 36
    f_cap = font(10)
    frames = []
    for t in range(0, T, stride):
        fr = Image.new("RGB", (2 * tile, 2 * (tile + cap)), SURF)
        dr = ImageDraw.Draw(fr)
        for i, (_, ims, act, text) in enumerate(eps):
            x, y = (i % 2) * tile, (i // 2) * (tile + cap)
            tt = min(t, len(act) - 1)
            fr.paste(Image.fromarray(np.asarray(ims[tt, 0])).resize((tile, tile), Image.BILINEAR), (x, y + cap))
            for j, line in enumerate(textwrap.wrap(text, 36)[:2]):
                dr.text((x + 6, y + 3 + j * 15), line, fill=INK, font=f_cap)
            if t >= len(act):
                dr.text((x + tile - 40, y + cap + 6), "done", fill="white", font=f_cap, stroke_width=2, stroke_fill="black")
        frames.append(fr.quantize(128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)


# ----------------------------------------------------------------------------------------------
ROLLOUTS = [("libero_spatial_t04_ep1_success", "spatial: bowl in drawer → plate"),
            ("libero_object_t02_ep0_success", "object: dressing → basket"),
            ("libero_goal_t07_ep0_success", "goal: turn on the stove"),
            ("libero_goal_t01_ep0_success", "goal: bowl → stove"),
            ("libero_10_t01_ep0_success", "long: cheese + butter → basket"),
            ("libero_goal_t00_ep0_fail", "goal: open the drawer")]


def rollouts_gif(video_dir, path, tile=192, stride=2):
    import imageio.v3 as iio
    vids = [(iio.imread(os.path.join(video_dir, n + ".mp4"))[:, :, :256], cap, n.endswith("success"))
            for n, cap in ROLLOUTS]
    T = max(len(v) for v, _, _ in vids)
    cap_h, cols = 22, 3
    rows = (len(vids) + cols - 1) // cols
    f_cap, f_tag = font(9), font(12, True)
    frames = []
    for t in list(range(0, T, stride)) + [T - 1] * 15:  # hold the final frame
        fr = Image.new("RGB", (cols * tile, rows * (tile + cap_h)), SURF)
        dr = ImageDraw.Draw(fr)
        for i, (v, cap, ok) in enumerate(vids):
            x, y = (i % cols) * tile, (i // cols) * (tile + cap_h)
            fr.paste(Image.fromarray(v[min(t, len(v) - 1)]).resize((tile, tile), Image.BILINEAR), (x, y + cap_h))
            dr.text((x + 5, y + 5), cap, fill=INK, font=f_cap)
            if t >= len(v) - 1:
                tag, col = ("success", GREEN) if ok else ("failed", "#d03b3b")
                dr.rectangle([x + tile - 70, y + cap_h + 6, x + tile - 6, y + cap_h + 24], fill=col)
                dr.text((x + tile - 64, y + cap_h + 8), tag, fill="white", font=f_tag)
        frames.append(fr.quantize(128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)


def loss_png(log, path):
    pts = [(int(m[1]), float(m[2])) for m in re.finditer(r"^step (\d+) loss ([\d.]+)", open(log).read(), re.M)]
    if not pts:
        print("no loss lines in", log); return
    st, lo = np.array(pts).T
    k = 9
    smooth = np.convolve(np.pad(lo, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120, facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.plot(st, lo, color=BLUE, lw=1, alpha=0.3)
    ax.plot(st, smooth, color=BLUE, lw=2)
    ax.annotate(f"{smooth[-1]:.3f}", (st[-1], smooth[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
    ax.set_title("Training L1 loss (normalized actions) — 2 GPUs, 16 samples/step", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("optimizer step", fontsize=9, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/libero")
    ap.add_argument("--log", default="runs_2k.log")
    ap.add_argument("--videos", default="runs/qwen3vl2b_libero_2k/eval_ckpt_002000/videos")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = Data(a.data)
    jobs = {"arch": lambda: architecture_svg(d, f"{OUT}/architecture.svg"),
            "io": lambda: model_io_gif(d, f"{OUT}/model_io.gif"),
            "demos": lambda: demos_gif(d, f"{OUT}/demos.gif"),
            "loss": lambda: loss_png(a.log, f"{OUT}/train_loss.png"),
            "rollouts": lambda: rollouts_gif(a.videos, f"{OUT}/rollouts.gif")}
    for k, fn in jobs.items():
        if not a.only or k in a.only.split(","):
            fn(); print("done", k)


if __name__ == "__main__":
    main()
