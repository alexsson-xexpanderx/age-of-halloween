"""The look test for Age of Halloween: a few of the new smooth models at game size and enlarged.

python tools/style_sheet.py [out_dir]   -> previews/style.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoh import graveyard as GY  # noqa: E402
from aoh import haunted as HT  # noqa: E402
from aoh import pumpkins as PK  # noqa: E402
from aoh import sculpt as S  # noqa: E402
from aoh import undead as UD  # noqa: E402
from aoh.animation import pose  # noqa: E402
from aoh.geometry import Part  # noqa: E402

GRASS = (78, 112, 52)
BUILDING = -45.0


def rgba(frame) -> np.ndarray:
    return frame.to_rgba(1).astype(np.float64) / 255


def on_grass(img: np.ndarray, size) -> Image.Image:
    h, w = img.shape[:2]
    rng = np.random.default_rng(3)
    bg = np.zeros((h, w, 3))
    bg[:] = np.array(GRASS) / 255
    bg *= (0.92 + 0.16 * rng.random((h, w, 1)))
    a = img[..., 3:]
    out = bg * (1 - a) + img[..., :3] * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def unit_frames(unit, action, t, headings):
    out = []
    for h in headings:
        p = pose(unit, action, t)
        cam = S.fit_camera(unit.root, h, p, scale=1.5 * unit.scale, pad=6)
        out.append(rgba(S.render(unit.root, h, p, cam)))
    return out


def static(root, heading=BUILDING):
    cam = S.fit_camera(root, heading, pad=6)
    return rgba(S.render(root, heading, camera=cam))


def row(images, gap=12):
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) - 1)
    out = np.zeros((h, w, 4))
    x = 0
    for i in images:
        out[h - i.shape[0]:, x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out


def main(out_dir: Path):
    t0 = time.time()
    tiles = []
    pac = PK.pacman()
    tiles.append(("Pac-Man, the scary pumpkin (walking, chomping)",
                  row(unit_frames(pac, "walk", 0.3, [-90, -135, 180, 135]) +
                      unit_frames(pac, "attack", 0.25, [-90]))))
    from aoh.colossus import walking_castle
    giant = walking_castle()
    giant.scale = 1.0  # at a quarter of his real size, to fit the sheet
    tiles.append(("The giant Pac-Man: a haunted castle at rest, and walking on its spider legs (a quarter of his size)",
                  row(unit_frames(giant, "idle", 0.0, [-90]) + unit_frames(giant, "walk", 0.0, [-90, -135]) +
                      unit_frames(giant, "walk", 0.5, [-45]))))
    mil = UD.militia()
    maa = UD.man_at_arms()
    tiles.append(("Skeleton Militia and Man-at-Arms (standing, walking, swinging)",
                  row(unit_frames(mil, "idle", 0.0, [-90, -135, 180]) + unit_frames(mil, "walk", 0.25, [-135]) +
                      unit_frames(mil, "attack", 0.5, [-90]) + unit_frames(maa, "idle", 0.0, [-90, -135, 180]) +
                      unit_frames(maa, "attack", 0.45, [-135]))))
    from aoh import estates
    tiles.append(("Haunted Victorian houses (West European)", row([static(estates.house("W", 2, v))
                                                                   for v in range(3)])))
    wall = Part("walls", children=[
        Part("a", offset=(0, 0, 0), children=[GY.wall_piece("x")]),
        Part("b", offset=(GY.TILE, 0, 0), children=[GY.wall_piece("x")]),
        Part("c", offset=(-GY.TILE * 0.5 - 1, 0, 0), children=[GY.wall_piece("post")]),
        Part("d", offset=(GY.TILE * 1.5 + 1, 0, 0), children=[GY.wall_piece("post")]),
    ])
    tiles.append(("Graveyard wall and dead trees", row([static(wall)] + [static(GY.dead_tree(70, v))
                                                                         for v in range(3)])))
    print(f"rendered in {time.time() - t0:.1f}s")

    pad = 16
    label_h = 18
    big = 2
    blocks = []
    for label, img in tiles:
        g1 = on_grass(img, None)
        g2 = g1.resize((g1.width * big, g1.height * big), Image.NEAREST)
        blocks.append((label, g1, g2))
    W = max(b[1].width + b[2].width for b in blocks) + 3 * pad
    H = sum(max(b[1].height, b[2].height) + label_h + pad for b in blocks) + pad
    sheet = Image.new("RGB", (W, H), (32, 26, 40))
    d = ImageDraw.Draw(sheet)
    y = pad
    for label, g1, g2 in blocks:
        d.text((pad, y), label + "   (left: game size, right: twice as big)", fill=(255, 200, 120))
        y += label_h
        sheet.paste(g1, (pad, y + max(0, g2.height - g1.height)))
        sheet.paste(g2, (2 * pad + g1.width, y))
        y += max(g1.height, g2.height) + pad
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out_dir / "style.png")
    print("wrote", out_dir / "style.png", sheet.size)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "previews")
