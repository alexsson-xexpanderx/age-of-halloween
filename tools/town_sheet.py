"""Preview sheets of Age of Halloween's buildings.

python tools/town_sheet.py [what ...]  -> previews/town_<what>.png
    what: houses, halls, military, economy, sites (default: all)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoh import estates as E  # noqa: E402
from aoh import sculpt as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "previews"
GRASS = np.array((70, 96, 50)) / 255
HEADING = -45.0
STYLES = [("G", 1, "Dark Age"), ("W", 2, "W Feudal"), ("W", 4, "W Imperial"), ("E", 3, "E Castle"),
          ("M", 3, "M Castle"), ("F", 3, "F Castle"), ("X", 2, "X Feudal"), ("X", 4, "X Imperial")]


def render(part):
    cam = S.fit_camera(part, HEADING, pad=4)
    img = S.render(part, HEADING, camera=cam).to_rgba(1).astype(np.float64) / 255
    rng = np.random.default_rng(1)
    bg = np.ones(img.shape[:2] + (3,)) * GRASS * (0.92 + 0.16 * rng.random(img.shape[:2] + (1,)))
    a = img[..., 3:]
    return Image.fromarray((np.clip(bg * (1 - a) + img[..., :3] * a, 0, 1) * 255).astype(np.uint8))


def grid(rows, path, title, scale=1.0):
    """rows: [(label, [(caption, PIL image)])]"""
    pad, lab = 8, 14
    W = max(sum(int(im.width * scale) + pad for _, im in cells) for _, cells in rows) + pad
    H = 24 + sum(max(int(im.height * scale) for _, im in cells) + lab * 2 + pad for _, cells in rows)
    out = Image.new("RGB", (W, H), (28, 22, 34))
    d = ImageDraw.Draw(out)
    d.text((pad, 4), title, fill=(255, 200, 120))
    y = 24
    for label, cells in rows:
        d.text((pad, y), label, fill=(230, 220, 240))
        y += lab
        h = max(int(im.height * scale) for _, im in cells)
        x = pad
        for cap, im in cells:
            im2 = im.resize((int(im.width * scale), int(im.height * scale)), Image.NEAREST) if scale != 1 else im
            out.paste(im2, (x, y + h - im2.height))
            d.text((x, y + h + 1), cap, fill=(180, 170, 190))
            x += im2.width + pad
        y += h + lab + pad
    out.save(path)
    return out.size


def main(whats):
    OUT.mkdir(exist_ok=True)
    sets = {
        "houses": [(f"House {v}", lambda s, a, v=v: E.house(s, a, v)) for v in range(3)],
        "halls": [("Town Center", E.town_center), ("Monastery", E.monastery), ("Market", E.market)],
        "military": [("Barracks", E.barracks), ("Archery Range", E.archery_range), ("Stable", E.stable),
                     ("Siege Workshop", E.siege_workshop)],
        "economy": [("Mill", lambda s, a: E.mill(s, a, 0.1)), ("Lumber Camp", E.lumber_camp),
                    ("Mining Camp", E.mining_camp), ("Blacksmith", lambda s, a: E.blacksmith(s, a, 0.3)),
                    ("University", lambda s, a: E.university(s, a, 0.2)), ("Dock", E.dock)],
    }
    from aoh import fortress as F
    sets["forts"] = [("Castle", F.castle), ("Keep", lambda s, a: F.watch_tower(s, 3)),
                     ("Watch tower", lambda s, a: F.watch_tower(s, 1)), ("Bombard tower", lambda s, a: F.watch_tower(s, 4)),
                     ("Gate tower", F.gate_tower)]
    for what in whats or list(sets) + ["sites", "walls"]:
        t = time.time()
        if what == "walls":
            from aoh.geometry import Part
            rows = []
            for kind in ("palisade", "stone", "fortified"):
                cells = []
                for s, a, lab in STYLES[1::2] + [("G", 1, "Dark Age")]:
                    if kind == "palisade" and s not in ("G",) and lab != "W Imperial":
                        continue
                    pieces = [Part(f"p{i}", offset=off, children=[F.wall_piece(kind, s, pc)]) for i, (pc, off) in
                              enumerate((("x", (-48.0, 0.0, 0.0)), ("post", (0.0, 0.0, 0.0)), ("x", (48.0, 0.0, 0.0)),
                                         ("y", (0.0, 48.0, 0.0)), ("v", (48.0, 48.0, 0.0))))]
                    cells.append((lab, render(Part("walls", children=pieces))))
                cells.append(("damaged", render(F.wall_piece(kind, "W", "x", damage=3))))
                cells.append(("building", render(F.wall_piece(kind, "W", "x", progress=0.4))))
                rows.append((kind, cells))
            gates = [(lab, render(Part("g", children=[F.gate_section(s, a if a >= 2 else 2, "x", False),
                                                        Part("t1", offset=(-68.0, 0, 0), children=[F.gate_tower(s, a)]),
                                                        Part("t2", offset=(68.0, 0, 0), children=[F.gate_tower(s, a)])])))
                     for s, a, lab in STYLES[1:6]]
            gates.append(("open", render(F.gate_section("E", 3, "x", True))))
            rows.append(("gates", gates))
            size = grid(rows, OUT / "town_walls.png", "Walls and gates")
        elif what == "sites":
            rows = [("Farm (dug, growing, ripe), fish trap, building sites, rubble",
                     [("farm 0", render(E.farm(0.0))), ("farm .5", render(E.farm(0.5))), ("farm 1", render(E.farm(1.0))),
                      ("fish trap", render(E.fish_trap(1.0))), ("site 2", render(E.construction(2, 1))),
                      ("site 3", render(E.construction(3, 3))), ("rubble 3", render(E.rubble(3, 0)))])]
            size = grid(rows, OUT / "town_sites.png", "Fields and sites")
        else:
            rows = []
            for name, fn in sets[what]:
                rows.append((name, [(lab, render(fn(s, a))) for s, a, lab in STYLES]))
            size = grid(rows, OUT / f"town_{what}.png", f"Buildings: {what} (each culture and age)")
        print(f"town_{what}.png {size} in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
