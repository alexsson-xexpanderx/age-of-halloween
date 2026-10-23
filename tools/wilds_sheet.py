"""Preview of the scenery: trees, stumps, mines, bushes, rocks, plants, decorations and projectiles, as props.py
builds them for the game. python tools/wilds_sheet.py -> previews/wilds.png"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aoh import props  # noqa: E402
from aoh.geometry import Part  # noqa: E402
from town_sheet import OUT, grid, render  # noqa: E402


def main():
    t = time.time()
    b = props.build
    rows = [
        ("Forests (oak, mixed, pine, snow, palm, jungle, bamboo)",
         [(f"{f} {v}", render(b({"model": "tree", "forest": f}, variant=v, count=8, fit={"height": 140})))
          for f in ("oak", "forest", "pine", "snow", "palm", "jungle", "bamboo") for v in (0, 1)]),
        ("Single trees, stumps", [(k, render(b({"model": "tree", "kind": k}, fit={"height": 130})))
                                  for k in ("birch", "acacia", "cherry", "dead")] +
         [("stump", render(b({"model": "stump", "kind": "oak"}))), ("birch stump", render(b({"model": "stump",
                                                                                              "kind": "birch"})))]),
        ("Resources", [("gold", render(b({"model": "ore", "kind": "gold"}, fit={"width": 90}))),
                       ("gold 2", render(b({"model": "ore", "kind": "gold"}, variant=2, fit={"width": 90}))),
                       ("stone", render(b({"model": "ore", "kind": "stone"}, fit={"width": 90}))),
                       ("berries", render(b({"model": "berry_bush"}))), ("rock", render(b({"model": "rock"},
                                                                                          fit={"width": 60}))),
                       ("plants", render(b({"model": "plants"}))), ("flowers", render(b({"model": "plants",
                                                                                         "flowers": True}))),
                       ("cactus", render(b({"model": "cactus"}))), ("haystack", render(b({"model": "haystack"})))]),
        ("Decorations", [(n, render(b({"model": "gaia", "name": n}))) for n in
                         ("yurt", "pavilion", "ruins", "stone_head", "statue", "graves", "heads", "signpost",
                          "standing_torch", "banner_flag", "relic", "skeleton", "broken_cart", "crater", "sea_rock")]),
        ("Projectiles", [(k, render(Part("p", offset=(0, 0, 20), children=[b({"model": "projectile", "kind": k},
                                                                                t=0.3)])))
                         for k in ("arrow", "fire_arrow", "bolt", "axe", "sword", "snowball", "cobble", "magma",
                                   "tnt")]),
    ]
    size = grid(rows, OUT / "wilds.png", "Scenery: trees, resources, decorations, projectiles")
    print("wilds.png", size, f"{time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
