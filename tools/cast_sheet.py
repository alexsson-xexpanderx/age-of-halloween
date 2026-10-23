"""Preview sheets of Age of Halloween's units: each unit standing in three directions, walking and attacking, at the
game's size and twice as big.

python tools/cast_sheet.py [group ...]   -> previews/cast_<group>.png  (groups: civilian foot uniques ...)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoh import sculpt as S  # noqa: E402
from aoh.animation import pose  # noqa: E402
from aoh.export import POSE_ACTION  # noqa: E402,F401

GRASS = np.array((78, 112, 52)) / 255
OUT = Path(__file__).resolve().parent.parent / "previews"


def frame(unit, action, t, heading):
    p = pose(unit, action, t)
    cam = S.fit_camera(unit.root, heading, p, scale=1.5 * unit.scale, pad=4)
    return S.render(unit.root, heading, p, cam).to_rgba(1).astype(np.float64) / 255


def on_grass(img):
    rng = np.random.default_rng(3)
    bg = np.ones(img.shape[:2] + (3,)) * GRASS * (0.92 + 0.16 * rng.random(img.shape[:2] + (1,)))
    a = img[..., 3:]
    return bg * (1 - a) + img[..., :3] * a


def strip(unit):
    poses = [("idle", 0.0, -90), ("idle", 0.0, -135), ("idle", 0.0, 180), ("walk", 0.25, -135),
             ("attack", 0.5, -90), ("attack", 0.5, -135)]
    if unit.carry:
        poses.append(("carry", 0.25, -135))
    imgs = [frame(unit, a, t, h) for a, t, h in poses]
    h = max(i.shape[0] for i in imgs)
    w = sum(i.shape[1] for i in imgs) + 6 * (len(imgs) - 1)
    out = np.zeros((h, w, 4))
    x = 0
    for i in imgs:
        out[h - i.shape[0]:, x:x + i.shape[1]] = i
        x += i.shape[1] + 6
    return out


def sheet(units, path: Path, title: str, big: int = 2):
    rows = []
    for u in units:
        s = on_grass(strip(u))
        im = Image.fromarray((np.clip(s, 0, 1) * 255).astype(np.uint8))
        rows.append((u, im, im.resize((im.width * big, im.height * big), Image.NEAREST)))
    pad, label = 10, 16
    W = max(r[1].width + r[2].width for r in rows) + 3 * pad
    H = sum(r[2].height + label + pad for r in rows) + pad + 20
    out = Image.new("RGB", (W, H), (30, 24, 36))
    d = ImageDraw.Draw(out)
    d.text((pad, 4), title + "   (stand S/SW/W, walk, attack x2" + (", carry" if any(r[0].carry for r in rows)
                                                                     else "") + "; left: game size, right: x2)",
           fill=(255, 200, 120))
    y = 24
    for u, im, big_im in rows:
        d.text((pad, y), f"{u.name}  (replaces {u.replaces}" + (f", {u.civ}" if u.civ else "") + ")",
               fill=(230, 220, 240))
        y += label
        out.paste(im, (pad, y + big_im.height - im.height))
        out.paste(big_im, (2 * pad + im.width, y))
        y += big_im.height + pad
    out.save(path)
    return out.size


def main(groups):
    from aoh.roster import HALLOWEEN
    OUT.mkdir(exist_ok=True)
    units = [f() for f in HALLOWEEN.values()]
    pick = {
        "civilian": lambda u: u.group == "civilian",
        "foot": lambda u: u.group == "foot" and not u.civ,
        "uniques": lambda u: u.civ is not None and u.group == "foot",
        "cavalry": lambda u: u.group == "cavalry",
        "siege": lambda u: u.group == "siege",
        "ships": lambda u: u.group == "ship",
        "animals": lambda u: u.group == "animal",
        "easter": lambda u: u.group == "easter",
    }
    titles = {"civilian": "Villagers, the Necromancer and the Vampire Lord", "foot": "Infantry and archers",
              "uniques": "Unique units on foot", "cavalry": "Cavalry", "siege": "Siege and the ghost train",
              "ships": "Ships", "animals": "Animals", "easter": "Easter eggs and the Dragon"}
    for g in groups or list(pick):
        t = time.time()
        chosen = [u for u in units if pick[g](u)]
        if not chosen:
            continue
        size = sheet(chosen, OUT / f"cast_{g}.png", titles[g])
        print(f"cast_{g}.png {size} {len(chosen)} units in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
