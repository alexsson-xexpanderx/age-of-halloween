"""The loading screen (interfac.drs 50163), as a Halloween title card.

A night sky with a huge full moon in the middle, the "AGE OF HALLOWEEN" title of the main menu at the top, and at
the bottom, black against the moon, a churchyard on a hill: a bare crooked tree, crosses and gravestones, an iron
fence and a haunted house with lit windows, jack-o'-lanterns glowing in the grass and bats crossing the moon. The game
writes its loading text in black in the middle of the screen: there the moon is pale and smooth, so it shows.

Only this screen uses its palette (50563), and the original is mostly greys: the mod gives it a palette of its own,
made from the new picture (`palette`), keeping the twenty Windows colours at 0-9 and 246-255 where they are.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import nightscape as N
from . import paint as P
from . import slp
from .palette import _lab

W, H = 800, 600
RESERVED = list(range(10)) + list(range(246, 256))  # the Windows colours: kept as they are
SKY_TOP, SKY_LOW = P.hexrgb("#07040e"), P.hexrgb("#4a2244")
MOON = (400, 330, 210)  # pale and smooth where the game writes its text
LOGO_W, LOGO_H, LOGO_Y = 600, 170, 6  # the title, at the top
INK = P.hexrgb("#0a060c")


def _churchyard(d, k) -> None:
    """The hill along the foot of the screen and everything on it, as one silhouette."""
    xs = np.linspace(0, W, 41)
    ys = 520 - 46 * np.exp(-((xs - 600) / 170) ** 2) - 20 * np.exp(-((xs - 150) / 140) ** 2) + 6 * np.sin(xs / 37)
    d.polygon([(x * k, y * k) for x, y in zip(xs, ys)] + [(W * k, H * k), (0, H * k)], fill=255)
    ground = lambda x: float(np.interp(x, xs, ys))
    N.silhouette_tree(d, 118, ground(118) + 4, 260, k, 9, width=1.5)
    N.silhouette_tree(d, 742, ground(742) + 4, 150, k, 10)
    # the house on the hill: a tall gable, a tower with a pointed roof, a crooked chimney
    hx, hy = 560, ground(600) + 6
    d.rectangle([hx * k, (hy - 92) * k, (hx + 96) * k, hy * k], fill=255)
    d.polygon([((hx - 8) * k, (hy - 90) * k), ((hx + 40) * k, (hy - 142) * k), ((hx + 88) * k, (hy - 90) * k)],
              fill=255)
    d.rectangle([(hx + 70) * k, (hy - 150) * k, (hx + 110) * k, hy * k], fill=255)
    d.polygon([((hx + 64) * k, (hy - 148) * k), ((hx + 90) * k, (hy - 214) * k), ((hx + 116) * k, (hy - 148) * k)],
              fill=255)
    d.polygon([((hx + 6) * k, (hy - 104) * k), ((hx + 16) * k, (hy - 132) * k), ((hx + 26) * k, (hy - 128) * k),
               ((hx + 20) * k, (hy - 100) * k)], fill=255)
    # gravestones and crosses, leaning
    for x, kind, h, lean in ((236, 0, 30, -0.15), (282, 1, 40, 0.1), (330, 0, 24, 0.2), (378, 1, 34, -0.08),
                             (430, 0, 28, 0.12), (470, 1, 30, -0.2), (60, 0, 26, 0.1), (196, 1, 26, 0.05)):
        g = ground(x) + 4
        if kind == 0:
            pts = [(-9, 0), (9, 0), (9, -h + 9), (0, -h), (-9, -h + 9)]
            pts = [(px + (-py) * lean, py) for px, py in pts]
            d.polygon([((x + px) * k, (g + py) * k) for px, py in pts], fill=255)
            d.ellipse([(x - 9 + h * lean) * k, (g - h - 1) * k, (x + 9 + h * lean) * k, (g - h + 17) * k], fill=255)
        else:
            top = (x + h * lean, g - h)
            d.line([(x * k, g * k), (top[0] * k, top[1] * k)], fill=255, width=int(5 * k))
            cy = g - h * 0.72
            cx = x + h * 0.72 * lean
            d.line([((cx - 11) * k, (cy + 2 * lean) * k), ((cx + 11) * k, (cy - 2 * lean) * k)], fill=255,
                   width=int(5 * k))
    for x in range(300, 520, 9):  # the iron fence before the house
        g = ground(x) + 2
        d.rectangle([x * k, (g - 26) * k, (x + 2) * k, g * k], fill=255)
        d.polygon([((x - 2) * k, (g - 25) * k), ((x + 1) * k, (g - 32) * k), ((x + 4) * k, (g - 25) * k)], fill=255)
    d.line([(298 * k, (ground(300) - 20) * k), (522 * k, (ground(520) - 20) * k)], fill=255, width=int(2 * k))
    d.line([(298 * k, (ground(300) - 8) * k), (522 * k, (ground(520) - 8) * k)], fill=255, width=int(2 * k))
    for x, y, s, f in ((300, 160, 16, 0.3), (350, 190, 11, -0.4), (470, 140, 13, 0.6), (520, 205, 9, 0.0),
                       (250, 230, 8, -0.2), (660, 120, 10, 0.5)):
        N.bat(d, x, y, s, k, f)


@lru_cache(maxsize=None)
def picture() -> np.ndarray:
    """The loading screen, RGB 0..1, 800x600."""
    from . import trinkets as T
    img = np.zeros((H, W, 3))
    N.sky(img, top=SKY_TOP, low=SKY_LOW, stars=320, seed=21, horizon=520)
    N.moon(img, *MOON, colour=(0.98, 0.93, 0.78), seas=0.07)
    for y, x0, x1, a in ((250, 150, 420, 0.25), (268, 380, 660, 0.2)):  # wisps of cloud
        m = N.shape(H, W, lambda d, k, y=y, x0=x0, x1=x1: d.ellipse([x0 * k, (y - 4) * k, x1 * k, (y + 4) * k],
                                                                      fill=255))
        N.lay(img, N._blur(m, 3) * a, P.hexrgb("#6a4a6a"))
    N.lay(img, N.shape(H, W, _churchyard), INK)
    hx, hy = 560, 520 - 46 * math.exp(-((600 - 600) / 170) ** 2) + 6 * math.sin(600 / 37) + 6
    for x, y, w, h in ((hx + 14, hy - 70, 12, 18), (hx + 50, hy - 70, 12, 18), (hx + 84, hy - 120, 10, 16),
                       (hx + 36, hy - 112, 9, 12)):  # the house's lit windows
        N.glow(img, x + w / 2, y + h / 2, 28, P.hexrgb("#ff9a2a"), 0.5)
        img[int(y):int(y + h), int(x):int(x + w)] = P.hexrgb("#ffc860")
        img[int(y + h / 2), int(x):int(x + w)] = INK
        img[int(y):int(y + h), int(x + w / 2)] = INK
    for seed, cx, foot, h in ((1, 160, 586, 46), (2, 268, 572, 34), (3, 404, 590, 40), (4, 690, 584, 30),
                              (5, 30, 594, 28)):
        jack = N.model(T.jack(seed, radius=3.0), height=h)
        N.glow(img, cx, foot - h / 2, h * 2.0, P.hexrgb("#ff9a2a"), 0.55)
        N.over(img, jack, int(cx - jack.shape[1] / 2), int(foot - jack.shape[0]))
    N.fog(img, 540, 600, colour=P.hexrgb("#3a3a5a"), strength=0.35, seed=7)
    mark = N.title(LOGO_W, LOGO_H)
    N.over(img, mark, (W - mark.shape[1]) // 2, LOGO_Y)
    return np.clip(img, 0, 1)


def palette(original: np.ndarray, rgb: np.ndarray = None) -> np.ndarray:
    """A palette for the picture: its 236 most telling colours (median cut) at 10-245, the Windows colours kept."""
    rgb = picture() if rgb is None else rgb
    flat = np.clip(rgb.reshape(-1, 3) * 255 + 0.5, 0, 255).astype(np.int64)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    boxes = [(colours, counts)]
    free = 256 - len(RESERVED)
    while len(boxes) < free:
        k = max(range(len(boxes)), key=lambda i: (np.ptp(boxes[i][0], axis=0).max() * np.sqrt(boxes[i][1].sum())
                                                   if len(boxes[i][0]) > 1 else -1))
        cols, cnt = boxes.pop(k)
        if len(cols) < 2:
            boxes.append((cols, cnt))
            break
        axis = int(np.argmax(np.ptp(cols, axis=0)))
        order = np.argsort(cols[:, axis], kind="stable")
        cols, cnt = cols[order], cnt[order]
        half = int(np.searchsorted(np.cumsum(cnt), cnt.sum() / 2)) + 1
        half = min(max(half, 1), len(cols) - 1)
        boxes += [(cols[:half], cnt[:half]), (cols[half:], cnt[half:])]
    out = np.asarray(original, np.int64)[:256].copy()
    free_idx = [i for i in range(256) if i not in RESERVED]
    for i, (cols, cnt) in zip(free_idx, boxes):
        out[i] = np.round((cols * cnt[:, None]).sum(0) / cnt.sum())
    for i in free_idx[len(boxes):]:
        out[i] = out[free_idx[0]]
    return out


def jasc(pal: np.ndarray) -> bytes:
    """A palette as the game stores it: JASC-PAL text."""
    lines = ["JASC-PAL", "0100", str(len(pal))] + [f"{r} {g} {b}" for r, g, b in np.asarray(pal, int)]
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def encode(original: bytes, original_palette: np.ndarray) -> tuple[bytes, np.ndarray]:
    """The new loading screen SLP (same frame and hotspot) and its palette."""
    frame = slp.decode(original)[0]
    rgb = picture()
    pal = palette(original_palette, rgb)
    free = np.array([i for i in range(256) if i not in RESERVED])
    flat = np.clip(rgb.reshape(-1, 3) * 255 + 0.5, 0, 255).astype(np.int64)
    keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    uniq, inverse = np.unique(keys, return_inverse=True)
    cols = _lab(np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], -1).astype(np.float64))
    lab = _lab(pal[free].astype(np.float64))
    best = free[((cols[:, None] - lab[None]) ** 2).sum(-1).argmin(1)]
    px = best[inverse].reshape(H, W).astype(np.int16)
    return slp.encode([slp.SlpFrame(px, frame.hotspot)], props=slp.frame_props(original)), pal
