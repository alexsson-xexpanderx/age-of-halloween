"""The eighteen wonders, haunted, and the scenario monuments. A wonder stands on a 5x5 footprint (half 112), each in
its civilisation's culture and after its old wonder (wonders.py): a haunted cathedral, a necropolis, a vampire's
castle, a ghost palace, a Day of the Dead church; the scenario pyramids are crypts (the player asked for no
pyramids).
"""
from __future__ import annotations

import math

import numpy as np

from . import fortress as F
from . import haunted as H
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from . import spooky as Y
from .geometry import Part
from .graveyard import skull
from .haunted import box, hull

HALF = 112.0


def _plinth(k, half=HALF - 6, h=6.0, steps=2) -> list:
    out = []
    for i in range(steps):
        a = 6.0 * i
        out.append(box((-half + a, -half + a, h * i), (half - a, half - a, h * (i + 1)), k.base))
    return out


def _yard(k, seed=0) -> list:
    """Graves, candles and pumpkins about the wonder's feet, on its camera sides."""
    out = []
    rng = np.random.default_rng(seed)
    for i in range(7):
        x, y = rng.uniform(40, 100), rng.uniform(-90, 100)
        if rng.random() < 0.5:
            x, y = y, x
        out += H.gravestone(x, y, 12.0, 7.0, i, seed=i)
    out += H.pumpkins(80.0, 90.0, 12.0, 3, seed) + H.candles(90.0, 70.0, 12.0, 5, seed)
    return out


def cathedral(style, spires=1, big_tower=False, seed=0) -> Part:
    """A haunted gothic cathedral: a long nave under a steep roof, a rose window, needle spires, a great tower."""
    k = H.kit(style, 4)
    out = _plinth(k)
    z = 12.0
    out += H.walls(k, -90, -40, 50, 30, z, 70, door=("+x", -5.0, 18.0, 40.0), every=18.0)
    out += H.gable(k, -90, 50, -40, 30, z + 70, 60.0, "x", over=3.0)
    out += H.walls(k, -40, -80, 0, 70, z, 64, door=("+y", -20.0, 14.0, 34.0), every=18.0, plinth=0.0)  # transept
    out += H.gable(k, -40, 0, -80, 70, z + 64, 44.0, "y", over=3.0)
    rose = S.Material(color=lambda l: np.where(
        (np.abs(np.sin(np.arctan2(l.p[:, 2] - (z + 90), l.p[:, 1] + 5) * 6)) > 0.3)[:, None], k.glow,
        P.hexrgb("#1a120e")), glow=lambda l: (np.abs(np.sin(np.arctan2(l.p[:, 2] - (z + 90), l.p[:, 1] + 5) * 6)) > 0.3)
        .astype(float))
    out.append(S.rod((50.0, -5.0, z + 90.0), (51.5, -5.0, z + 90.0), 12.0, rose))
    if spires >= 2:
        for y in (-34.0, 24.0):
            out += H.walls(k, 34, y - 10, 54, y + 10, z, 110, every=10.0, plinth=0.0)
            out += H.spire(44.0, y, z + 110, 13.0, 80.0, k.roof)
    else:
        out += H.walls(k, -110, -20, -76, 14, z, 130, every=12.0, plinth=0.0)
        out += H.spire(-93.0, -3.0, z + 130, 22.0, 90.0, k.roof)
        for sx, sy in ((-108, -18), (-78, -18), (-108, 12), (-78, 12)):
            out += H.spire(float(sx), float(sy), z + 130, 3.0, 26.0, k.roof, finial=False)
    if big_tower:
        out += H.walls(k, -34, -34, -6, -6, z + 64, 60, every=10.0, plinth=0.0) + H.spire(-20.0, -20.0, z + 124, 18.0,
                                                                                           60.0, k.roof)
    out += H.bats([(20.0, 40.0, 190.0), (-40.0, 0.0, 230.0), (60.0, -30.0, 210.0)], seed)
    out += _yard(k, seed) + H.team_flag((48.0, 28.0, z + 70), 24.0, seed)
    return Part("wonder", boxes=out)


def dracula(style="E", seed=0) -> Part:
    """A vampire lord's castle on a crag: a tall keep crowned with needle spires, red windows, a bat swarm."""
    k = H.kit(style, 4)
    out = []
    rock = P.mottled(P.hexrgb("#3e3a44"), P.hexrgb("#141218"), 4.0, 0.6, seed)
    for i, (r, h) in enumerate(((104.0, 18.0), (86.0, 34.0), (70.0, 46.0))):
        out.append(hull([(math.cos(a) * r * (1 + 0.1 * math.sin(a * 5 + i)), math.sin(a) * r * (1 + 0.1 * math.cos(a * 3)),
                          z) for a in np.linspace(0, 2 * math.pi, 9)[:-1] for z in (0.0, h)], rock))
    z = 46.0
    out += H.walls(k, -50, -50, 30, 30, z, 70, door=("+y", -10.0, 14.0, 30.0), m=F._stone("E", "fortified"))
    out += [box((-52, -52, z + 70), (32, 32, z + 74), k.base)]
    out += H.spire(-10.0, -10.0, z + 74, 30.0, 110.0, k.roof)
    for x, y in ((-50.0, -50.0), (30.0, -50.0), (-50.0, 30.0), (30.0, 30.0)):
        out += [S.rod((x, y, z), (x, y, z + 100.0), 11.0, F._stone("E", "fortified"))] + H.spire(x, y, z + 100.0, 12.0,
                                                                                                  60.0, k.roof)
    out += H.bats([(x, y, z + 160 + 10 * i) for i, (x, y) in enumerate(((0, 40), (-40, 10), (40, -20), (10, -40),
                                                                         (-20, 50), (50, 30)))], seed)
    out += H.team_flag((-10.0, -10.0, z + 190.0), 24.0, seed)
    return Part("wonder", boxes=out)


def necropolis(style="M", minarets=2, seed=0) -> Part:
    """A great domed tomb: a cube of sandstone banded with hieroglyphs under a huge onion dome, minarets, mummy cases
    at the door, a green glow from within."""
    k = H.kit(style, 4)
    out = _plinth(k, steps=3)
    z = 18.0
    out += H.walls(k, -70, -70, 50, 50, z, 64, door=("+y", -10.0, 20.0, 40.0))
    out += [box((-71, -71, z + 44), (51, 51, z + 52), k.upper)]
    out += H.flat_roof(k, -70, -70, 50, 50, z + 64)
    out += H.dome(-10.0, -10.0, z + 66, 50.0, k.roof, onion=True)
    for i in range(minarets):
        x, y = [(70.0, 70.0), (-90.0, 70.0), (70.0, -90.0), (-90.0, -90.0)][i]
        out += H.tower(k, x, y, 9.0, z, 120.0, seed=i)
    out += H.sarcophagus(4.0, 54.0, z, "y") + H.sarcophagus(-24.0, 54.0, z, "y")
    out += H.urn(60.0, 20.0, z, 8.0) + H.urn(60.0, -20.0, z, 8.0)
    return Part("wonder", boxes=out)


def pyramid(style="M", seed=0, smooth=True, half=100.0, cap=P.GOLD) -> Part:
    """A pyramid: smooth limestone with a gold cap, a dark doorway glowing green halfway up its face."""
    k = H.kit(style, 4)
    h = half * 1.25
    stone = P.bricks(P.hexrgb("#c8b890"), P.hexrgb("#7e6e4e"), 9.0, 6.0, seed=seed) if not smooth else \
        P.mottled(P.hexrgb("#c8b890"), P.hexrgb("#7e6e4e"), 8.0, 0.4, seed)
    out = [hull([(-half, -half, 0), (half, -half, 0), (-half, half, 0), (half, half, 0), (0, 0, h)], stone)]
    out.append(hull([(-half * 0.12, -half * 0.12, h * 0.88), (half * 0.12, -half * 0.12, h * 0.88),
                     (-half * 0.12, half * 0.12, h * 0.88), (half * 0.12, half * 0.12, h * 0.88), (0, 0, h + 0.5)],
                    P.metal(cap, shine=0.9)))
    door = hull([(-6.0, half * 0.62 + 0.4, h * 0.25), (6.0, half * 0.62 + 0.4, h * 0.25), (0, half * 0.55 + 0.4,
                                                                                       h * 0.38),
                 (-6.0, half * 0.62 + 1.6, h * 0.25), (6.0, half * 0.62 + 1.6, h * 0.25)], S.mat(P.SLIME, glow=0.9))
    out.append(door)
    return Part("monument", boxes=out)


def step_pyramid(style="X", shrines=1, steps=6, seed=0, comb=False) -> Part:
    """A Day of the Dead step pyramid: painted terraces, a stair up the front lined with candles and marigolds, a
    temple (or two) on top with a fire, a rack of skulls at its foot."""
    k = H.kit(style, 4)
    out = []
    half = HALF - 8
    sh = 18.0
    for i in range(steps):
        a = (half - 30.0) / steps * i
        out.append(box((-half + a, -half + a, sh * i), (half - a, half - a, sh * (i + 1)), k.wall if i % 2 == 0 else
                       k.base))
    top = sh * steps
    stair = P.bricks(P.hexrgb("#8a8274"), P.hexrgb("#4a4438"), 6.0, 3.0, seed=seed)
    out.append(hull([(-14, half, 0), (14, half, 0), (-14, half - 30 - 10, top), (14, half - 30 - 10, top),
                     (-14, half - 30 - 10, 0), (14, half - 30 - 10, 0)], stair))
    a = half - 30.0
    for j in range(shrines):
        cx = 0.0 if shrines == 1 else (-16.0 if j == 0 else 16.0)
        w = 22.0 if shrines == 1 else 14.0
        out += H.walls(k, cx - w, -w, cx + w, w, top, 26, door=("+y", cx, 8.0, 16.0), plinth=0.0, every=16.0)
        out += H.terrace(k, cx - w, -w, cx + w, w, top + 26, steps=2, step=4.0, inset=4.0)
        f = S.Ellipsoid(S.mat(P.hexrgb("#ffb030"), glow=1.0), S._orient((cx, 0.0, top + 40.0)), radii=(5.0, 5.0, 8.0))
        f.casts = False
        out.append(f)
    if comb:
        out.append(box((-18.0, -6.0, top + 34.0), (18.0, 2.0, top + 60.0), k.wall))
    for i in range(6):
        out += H.candles(-18.0 if i % 2 else 18.0, half - 4 - i * 7, sh * i * 0.6, 2, seed + i)
    out += H.marigolds(-30.0, half + 4.0, 0.0, 10, seed) + H.marigolds(30.0, half + 4.0, 0.0, 10, seed + 1)
    for i in range(6):
        out += skull((60.0 + i * 7.0, half + 6.0, 4.0), 2.4, i) + skull((60.0 + i * 7.0, half + 6.0, 11.0), 2.4, i + 6)
    out += H.team_flag((0.0, 0.0, top + 50.0), 24.0, seed)
    return Part("wonder", boxes=out)


def cursed_pagoda(style="F", tiers=5, seed=0, round_hall=False) -> Part:
    """A cursed pagoda: tier on tier of curved black roofs hung with red lanterns, a spire of rings, a torii gate."""
    k = H.kit(style, 4)
    out = _plinth(k, steps=2)
    z = 12.0
    r = 66.0
    th = 30.0
    for i in range(tiers):
        rr = r * (1 - 0.13 * i)
        if round_hall:
            out.append(S.rod((-10, -10, z + th * i), (-10, -10, z + th * (i + 1) - 6), rr * 0.8, k.wall))
            out.append(S.turned((-10, -10, z + th * (i + 1) - 6), [(0, 0), (rr * 1.15, 0), (rr * 0.8, 6.0), (0.1, 14.0)],
                                k.roof))
        else:
            out += H.walls(k, -10 - rr, -10 - rr, -10 + rr, -10 + rr, z + th * i, th - 8, plinth=0.0,
                           door=("+y", -10.0, 14.0, 18.0) if i == 0 else None, every=16.0)
            out += H.pagoda_roof(k, -10 - rr, -10 + rr, -10 - rr, -10 + rr, z + th * (i + 1) - 8, 12.0, over=12.0)
        out += H.lanterns_hanging([(-10 + rr + 9, -10 + rr * 0.5, z + th * (i + 1) - 12),
                                   (-10 + rr * 0.5, -10 + rr + 9, z + th * (i + 1) - 12)], k)
    top = z + th * tiers
    gold = P.metal(P.GOLD, shine=0.8)
    out.append(S.rod((-10, -10, top), (-10, -10, top + 50), 1.4, gold))
    for i in range(6):
        out.append(S.torus((-10, -10, top + 8 + i * 6.0), 4.0 - i * 0.4, 0.6, gold))
    red = S.mat(P.hexrgb("#8a1a14"), shine=0.4, vivid=0.7)
    out += [S.rod((60.0, 80.0, 0.0), (60.0, 80.0, 40.0), 2.0, red), S.rod((96.0, 80.0, 0.0), (96.0, 80.0, 40.0), 2.0, red),
            box((52.0, 78.0, 40.0), (104.0, 82.0, 44.0), S.mat(P.hexrgb("#1a1414"))),
            box((56.0, 78.6, 32.0), (100.0, 81.4, 34.5), red)]
    return Part("wonder", boxes=out)


def great_arch(style="M", seed=0) -> Part:
    """A great ruined arch of brick between arcaded wings, ghost lights in its windows."""
    k = H.kit(style, 4)
    out = _plinth(k)
    z = 12.0
    brick = P.bricks(P.hexrgb("#9a7e5a"), P.hexrgb("#4a3a28"), 3.0, 1.8, seed=seed)
    out += H.walls(k, -100, -20, -36, 10, z, 70, m=brick, every=12.0) + H.walls(k, 24, -20, 88, 10, z, 70, m=brick,
                                                                               every=12.0)
    arch = S.turned((-6.0, -5.0, z), [(30.0, 0), (36.0, 0), (36.0, 70.0), (30.0, 70.0)], brick)
    out.append(box((-36, -20, z), (24, 10, z + 120), H.facade(brick, [H.Opening("+y", -26.0, 14.0, z, z + 92.0, "round")],
                                                                     k)))
    out += H.flat_roof(k, -100, -20, 88, 10, z + 70) + H.flat_roof(k, -36, -20, 24, 10, z + 120)
    return Part("wonder", boxes=out)


def stave_church(style="E", seed=0) -> Part:
    """A black stave church: storeys under steep shingle gables crossing each other, a dragon's head rearing from
    every gable end the camera sees, a needle spire, ravens and webs."""
    k = H.kit(style, 4)
    out = _plinth(k)
    z = 12.0
    horn = S.mat(P.hexrgb("#1a1414"), shine=0.4)
    for i, (r, h) in enumerate(((66.0, 34.0), (48.0, 28.0), (32.0, 26.0), (20.0, 20.0))):
        out += Y.storey(k, -r, -r, r, r, z, h, door=("+y", 0.0, 12.0, 22.0) if i == 0 else None, every=16.0,
                        seed=seed + i, boarded=0.2, shutters=0.2)
        axis = "x" if i % 2 == 0 else "y"
        out += Y.roof(k, -r, r, -r, r, z + h, "gable", axis, rise=r * 0.85, over=6.0, seed=seed + i)[0]
        peak = (r + 6.0, 0.0, z + h + r * 0.85) if axis == "x" else (0.0, r + 6.0, z + h + r * 0.85)
        p = np.asarray(peak, float)
        d = np.array([1.0, 0, 0]) if axis == "x" else np.array([0, 1.0, 0])
        out += [S.rod(tuple(p), tuple(p + d * 4.0 + (0, 0, 6.0)), 1.6, horn, 1.0),
                S.Ellipsoid(horn, S._orient(tuple(p + d * 6.5 + (0, 0, 7.0))), radii=(2.0, 2.0, 1.4)),
                S.ball(tuple(p + d * 7.6 + (0, 0, 7.6)), 0.6, S.mat(P.hexrgb("#ff2410"), glow=1.0))]
        z += h + r * 0.4
    out += H.spire(0.0, 0.0, z, 12.0, 54.0, k.roof)
    out += Y.ravens([(66.0 + 6.0, 0.0, 12.0 + 34.0 + 66.0 * 0.85 + 1.0)], seed)
    out += Y.web((66.0, 66.0, 45.5), (-1.0, 0, 0), (0, 0, -1.0), 10.0, seed)
    out += _yard(k, seed)
    return Part("wonder", boxes=out)


def round_tower_ruin(style="W", seed=0) -> Part:
    """A ruined church on a crag beside a tall round tower, a glow in its empty windows."""
    k = H.kit(style, 4)
    rock = P.mottled(P.hexrgb("#4a4650"), P.hexrgb("#1a181e"), 4.0, 0.6, seed)
    out = [hull([(math.cos(a) * 104, math.sin(a) * 104, z) for a in np.linspace(0, 2 * math.pi, 9)[:-1]
                 for z in (0.0, 24.0)], rock)]
    z = 24.0
    stone = F._stone("W", "fortified")
    out += H.walls(k, -80, -40, 40, 30, z, 60, m=stone, door=("+x", -5.0, 14.0, 30.0))
    for x in np.linspace(-70, 30, 5):  # a broken roof: ribs only
        out.append(hull([(x, -42, z + 60), (x + 4, -42, z + 60), (x, -5, z + 100), (x + 4, -5, z + 100),
                         (x, 32, z + 60), (x + 4, 32, z + 60)], H.kit("W", 4).trim))
    out += [S.rod((60.0, 50.0, 0.0), (60.0, 50.0, 190.0), 16.0, stone, 13.0),
            S.turned((60.0, 50.0, 190.0), [(0, 0), (14.0, 0), (0.1, 30.0)], k.roof)]
    out += _yard(k, seed)
    return Part("wonder", boxes=out)


def octagon(style="E", seed=0) -> Part:
    """A dark octagonal chapel under a domed tower (after Aachen), red glass glowing."""
    k = H.kit(style, 4)
    out = _plinth(k)
    z = 12.0
    stone = F._stone("E", "fortified")
    out.append(S.rod((0, 0, z), (0, 0, z + 80), 70.0, stone))
    for a in range(8):
        ang = a * math.pi / 4 + math.pi / 8
        out.append(box((math.cos(ang) * 68 - 3, math.sin(ang) * 68 - 3, z + 30), (math.cos(ang) * 68 + 3,
                                                                                   math.sin(ang) * 68 + 3, z + 60),
                       S.mat(k.glow, glow=1.0)))
    out += H.dome(0.0, 0.0, z + 80, 60.0, k.roof)
    out += H.walls(k, 60, -26, 104, 26, z, 120, plinth=0.0, every=12.0) + H.spire(82.0, 0.0, z + 120, 20.0, 60.0, k.roof)
    return Part("wonder", boxes=out)


def stupa(style="F", seed=0) -> Part:
    """A white stupa gone grey with age, a gold spire, prayer flags in the owner's colour, butter lamps."""
    k = H.kit(style, 4)
    out = _plinth(k, steps=3)
    z = 18.0
    white = P.mottled(P.hexrgb("#b8b4a8"), P.hexrgb("#5a5850"), 5.0, 0.5, seed)
    out.append(S.ball((0, 0, z), (80.0, 80.0, 70.0), white).clip((0, 0, -1), -z))
    out.append(box((-20, -20, z + 66), (20, 20, z + 82), white))
    gold = P.metal(P.GOLD, shine=0.8)
    out.append(S.rod((0, 0, z + 82), (0, 0, z + 150), 16.0, gold, 1.0))
    for i in range(8):
        out.append(S.torus((0, 0, z + 86 + i * 7.0), 15.0 - i * 1.8, 1.0, gold))
    out += H.candles(70.0, 70.0, 12.0, 6, seed) + H.candles(80.0, 30.0, 12.0, 6, seed + 1)
    return Part("wonder", boxes=out)


def golden_tower(style="X", seed=0) -> Part:
    """A many-sided tower of tarnished gold (after the Torre del Oro), a ghost light on top."""
    k = H.kit("W", 4)
    gold = P.metal(P.hexrgb("#a88a3a"), rust=0.6, seed=seed)
    out = _plinth(k)
    z = 12.0
    out.append(S.rod((0, 0, z), (0, 0, z + 110), 60.0, gold, 56.0))
    out += F._crown(k, "M", 40.0, z + 110, gold)
    out.append(S.rod((0, 0, z + 110), (0, 0, z + 160), 30.0, gold, 26.0))
    out += H.dome(0.0, 0.0, z + 160, 24.0, P.metal(P.GOLD, shine=0.9))
    glow = S.ball((0, 0, z + 200), 6.0, S.mat(P.SLIME, glow=1.0))
    glow.casts = False
    out.append(glow)
    return Part("wonder", boxes=out)


def great_hall(style="E", seed=0) -> Part:
    """A great haunted hall: two storeys of black stone under a steep, sagging gable, dormers, a chimney, two
    turrets with needle spires, horse skulls on its gable ends, a porch and a graveyard."""
    k = H.kit("E", 4)
    out = _plinth(k)
    hall, top = Y.manor(k, -90, -56, 60, 40, 2, "x", "+y", door_at=-20.0, dormers=3, chimney_at=(-70.0, -36.0),
                        seed=seed, storey_h=24.0, z0=12.0, tower=(66.0, 50.0, 12.0, 120.0))
    out += hall + Y.turret(k, -84.0, 50.0, 10.0, 12.0, 100.0, seed=seed + 1)
    for x in (-96.0, 66.0):
        out.append(S.Ellipsoid(P.bone(seed), S._orient((x, -8.0, top - 30.0)), radii=(4.0, 6.0, 10.0)))
    out += _yard(k, seed)
    return Part("wonder", boxes=out)


WONDERS = {
    "B": lambda: cathedral("W", spires=1, seed=1),          # Britons: a gothic cathedral with one great tower
    "R": lambda: cathedral("W", spires=2, seed=2),          # Franks: twin spires
    "H": lambda: great_hall("E", 3),                        # Goths: a haunted great hall
    "U": lambda: octagon("E", 4),                           # Teutons
    "I": lambda: stave_church("E", 5),                      # Vikings
    "L": lambda: round_tower_ruin("W", 6),                  # Celts
    "Y": lambda: necropolis("M", 4, 7),                     # Byzantines
    "P": lambda: great_arch("M", 8),                        # Persians
    "S": lambda: necropolis("M", 1, 9),                     # Saracens
    "T": lambda: necropolis("M", 4, 10),                    # Turks
    "J": lambda: ghost_palace("F", 3, 11),                  # Japanese: a ghost palace keep
    "Z": lambda: ghost_palace("F", 2, 12, wings=True),      # Chinese: a palace with wings
    "N": lambda: stupa("F", 13),                            # Mongols
    "K": lambda: ghost_palace("F", 4, 14),                  # Koreans: a tall, slender keep
    "C": lambda: golden_tower("X", 15),                     # Spanish
    "G": lambda: dracula("E", 16),                          # Huns
    "A": lambda: dead_cathedral("X", 2, 17),                # Aztecs: a Day of the Dead church, twin towers
    "M": lambda: dead_cathedral("X", 1, 18),                # Mayans: one tower and a dome (no pyramids)
}


def _wisps(points, seed=0) -> list:
    """Will-o'-wisps: little blue ghost flames floating in the air (they cast nothing)."""
    out = []
    for p in points:
        w = S.Ellipsoid(S.mat(P.hexrgb("#7ab8ff"), glow=1.0, vivid=1.0), S._orient(tuple(p)), radii=(1.6, 1.6, 2.6))
        w.casts = False
        out.append(w)
    return out


def dead_cathedral(style="X", towers=2, seed=0) -> Part:
    """A Day of the Dead church: a painted nave under a terracotta roof, a tall front with a great arched door
    framed in marigolds and a rose window, bell towers (two, or one and a dome over the crossing), skulls along
    its top, papel picado, candles and a graveyard of painted crosses."""
    k = H.kit(style, 4)
    out = _plinth(k)
    z = 12.0
    nave, top = Y.manor(k, -86, -32, 30, 24, 3, "x", "+x", roof_kind="tiles", porch_at=False, dormers=0, seed=seed,
                        storey_h=22.0, z0=z, webs=False)
    out += nave
    fz = z + 2.5
    front = H.facade(k.upper, [H.Opening("+x", -16.0, 8.0, fz, fz + 34.0, "door"),
                               H.Opening("+x", -10.0, 2.0, fz + 44.0, fz + 62.0, "round")], k)
    out.append(box((30.0, -36.0, fz), (42.0, 28.0, fz + 74.0), front))
    out += Y.roof(k, 30.0, 42.0, -36.0, 28.0, fz + 74.0, "tiles", "y", rise=16.0, over=2.0, seed=seed)[0]
    for i in range(9):  # skulls along the top of the front
        out += skull((42.5, -32.0 + i * 7.0, fz + 74.6), 2.0, i)
    for i in range(13):  # a marigold arch round the door
        a = math.pi * i / 12
        out.append(S.ball((43.2, -4.0 - math.cos(a) * 15.0, fz + 22.0 + math.sin(a) * 15.0), 2.0,
                          S.mat([P.hexrgb("#f08a10"), P.hexrgb("#f0b020")][i % 2], vivid=1.0)))
    if towers == 2:
        for y in (-46.0, 38.0):
            out += Y.turret(k, 36.0, y, 11.0, z, 110.0, seed=seed + int(y))
    else:
        out += Y.turret(k, -80.0, 32.0, 12.0, z, 120.0, seed=seed)
        out += H.dome(-30.0, -4.0, top - 6.0, 22.0, k.roof, finial=P.hexrgb("#2a2628"))
    from .estates import papel_picado
    out += papel_picado((48.0, -46.0, 40.0), (48.0, 38.0, 40.0), seed) + papel_picado((70.0, -60.0, 30.0),
                                                                                      (70.0, 60.0, 30.0), seed + 1)
    out += H.marigolds(54.0, -20.0, 12.0, 10, seed) + H.marigolds(54.0, 14.0, 12.0, 10, seed + 1)
    out += H.candles(50.0, -6.0, 12.0, 6, seed) + _yard(k, seed)
    return Part("wonder", boxes=out)


def ghost_palace(style="F", tiers=3, seed=0, wings=False) -> Part:
    """A haunted palace keep: storey on storey of black timber and torn paper walls, each under its own curved
    gable (the gables turning a quarter each storey, swept up at the eaves, never a pyramid), red lanterns, blue
    will-o'-wisps floating round it, a spire of rings, a torii gate; with low wings for the larger palaces."""
    k = H.kit(style, 4)
    out = _plinth(k, steps=2)
    z = 12.0
    r0 = 62.0 if not wings else 50.0
    for i in range(tiers):
        rr = r0 * (1 - 0.17 * i)
        th = 26.0 - 2.0 * i
        out += Y.storey(k, -10 - rr, -10 - rr, -10 + rr, -10 + rr, z, th, door=("+y", -10.0, 14.0, 18.0) if i == 0 else
                        None, every=16.0, seed=seed + i, boarded=0.12, shutters=0)
        out += Y.roof(k, -10 - rr, -10 + rr, -10 - rr, -10 + rr, z + th, "curved", "x" if i % 2 == 0 else "y",
                      rise=rr * 0.42, over=10.0, seed=seed + i)[0]
        out += H.lanterns_hanging([(-10 + rr + 10, -10 + rr * 0.5, z + th - 3), (-10 + rr * 0.5, -10 + rr + 10, z + th - 3)],
                                  k)
        z += th + rr * 0.2
    top = z + 10.0
    gold = P.metal(P.GOLD, shine=0.8, rust=0.3)
    out.append(S.rod((-10, -10, top - 12.0), (-10, -10, top + 40), 1.4, gold))
    for i in range(6):
        out.append(S.torus((-10, -10, top + 2 + i * 6.0), 4.0 - i * 0.4, 0.6, gold))
    if wings:
        for (x0, y0, x1, y1, ax, face) in ((-104, 44, -40, 96, "x", "+y"), (44, -104, 96, -40, "y", "+x")):
            wing, _ = Y.manor(k, x0, y0, x1, y1, 1, ax, face, porch_at=False, dormers=0, seed=seed + 9, z0=12.0,
                              storey_h=18.0)
            out += wing
    red = S.mat(P.hexrgb("#8a1a14"), shine=0.4, vivid=0.7)
    out += [S.rod((60.0, 86.0, 0.0), (60.0, 86.0, 40.0), 2.0, red), S.rod((96.0, 86.0, 0.0), (96.0, 86.0, 40.0), 2.0, red),
            box((52.0, 84.0, 40.0), (104.0, 88.0, 44.0), S.mat(P.hexrgb("#1a1414"))),
            box((56.0, 84.6, 32.0), (100.0, 87.4, 34.5), red)]
    out += Y.web((96.0, 86.0, 38.0), (-1.0, 0, 0), (0, 0, -1.0), 10.0, seed)
    out += _wisps([(60.0, 40.0, 60.0), (40.0, 70.0, 80.0), (80.0, 10.0, 90.0), (-60.0, 70.0, 70.0)], seed)
    out += Y.ravens([(60.0, 86.0, 44.4), (100.0, 86.0, 44.4)], seed)
    return Part("wonder", boxes=out)


def crypt(half: float = 80.0, seed: int = 0) -> Part:
    """A great crypt (in place of the scenario pyramids): a dark stone mausoleum with a portico of columns under a
    pediment, a heavy door with a green glow round it, urns, a weeping angel, an iron fence, graves and a dead
    tree."""
    k = H.kit("E", 4)
    stone = P.bricks(P.hexrgb("#6a6672"), P.hexrgb("#2a2830"), 6.0, 3.0, moss=0.8, seed=seed)
    out = [box((-half, -half, 0), (half, half, 4.0), k.base)]
    w, d = half * 0.5, half * 0.42
    h = half * 0.5
    door = H.Opening("+y", -w * 0.22, w * 0.22, 6.0, 6.0 + h * 0.62, "round")
    out.append(box((-w, -d - w * 0.3, 4.0), (w, d, 4.0 + h), H.facade(stone, [door], H.Kit(**{**k.__dict__,
                                                                                             "glow": P.hexrgb("#ff7a2a")}))))
    out.append(box((-w - 2, -d - w * 0.3 - 2, 4.0 + h), (w + 2, d + w * 0.42 + 2, 4.0 + h + 3.0), k.base))  # entablature
    out += Y.roof(k, -w - 2, w + 2, -d - w * 0.3 - 2, d + w * 0.42 + 2, 7.0 + h, "gable", "y", rise=w * 0.55, over=2.0,
                  seed=seed, crest=False)[0]
    col = P.mottled(P.hexrgb("#7a7684"), P.hexrgb("#3a3644"), 2.0, 0.4, seed)
    for i in range(4):  # the portico's columns
        x = -w + w * 2 * (i + 0.5) / 4
        out.append(S.turned((x, d + w * 0.32, 4.0), [(0, 0), (3.4, 0), (2.4, 2.0), (2.1, h - 2.0), (3.2, h), (0, h)], col))
    out.append(box((-w - 1, d, 0.0), (w + 1, d + w * 0.42 + 1, 4.0), k.base))
    for x in (-w * 0.65, w * 0.65):
        out += H.urn(x, d + w * 0.62, 4.0, h * 0.25)
    out += Y.web((w - w / 4 * 1.0, d + w * 0.32, 4.0 + h - 0.5), (-1.0, 0, 0), (0, 0, -1.0), w * 0.35, seed)
    out += Y.fence((-half + 6, half - 6, 4.0), (half - 6, half - 6, 4.0), 9.0, seed) + \
        Y.fence((half - 6, -half + 6, 4.0), (half - 6, half - 6, 4.0), 9.0, seed + 1)
    rng = np.random.default_rng(seed)
    out += Y.graves([(rng.uniform(w + 8, half - 12), rng.uniform(-half + 12, half - 12)) for _ in range(5)] +
                    [(rng.uniform(-half + 12, w), rng.uniform(d + w * 0.8, half - 12)) for _ in range(3)], seed)
    out += Y.ravens([(0.0, d + w * 0.42 + 2, 7.0 + h + w * 0.55 + 1.0)], seed)
    return Part("monument", boxes=out).add(Y.dead_tree(half * 0.7, -half * 0.6, half * 0.8, seed % 5))


def wonder(letter: str) -> Part:
    return WONDERS.get(letter, WONDERS["B"])()


# --------------------------------------------------------------------------- scenario monuments

def dome_of_the_rock() -> Part:
    k = H.kit("M", 4)
    out = [box((-80, -80, 0), (80, 80, 8.0), k.base)]
    blue = P.mottled(P.hexrgb("#2a5a7a"), P.hexrgb("#10283a"), 3.0, 0.4)
    out.append(S.rod((0, 0, 8.0), (0, 0, 50.0), 70.0, H.facade(blue, [H.Opening("+y", -6.0, 6.0, 20.0, 40.0, "round"),
                                                                         H.Opening("+x", -6.0, 6.0, 20.0, 40.0, "round")],
                                                               k)))
    out += H.dome(0.0, 0.0, 50.0, 40.0, P.metal(P.GOLD, rust=0.5, shine=0.9))
    return Part("monument", boxes=out)


def small_pyramid() -> Part:
    return crypt(80.0, 1)


def large_pyramid() -> Part:
    return crypt(128.0, 2)


def cathedral_monument() -> Part:
    return cathedral("W", spires=2, seed=20)


def mosque(minarets: int = 2) -> Part:
    return necropolis("M", minarets, 21)


def tower_of_flies() -> Part:
    """A tall lighthouse tower whose lamp burns green."""
    k = H.kit("W", 4)
    stone = F._stone("W", "fortified")
    out = [box((-60, -60, 0), (60, 60, 4.0), k.base), S.rod((0, 0, 4.0), (0, 0, 180.0), 40.0, stone, 30.0)]
    lamp = S.rod((0, 0, 180.0), (0, 0, 200.0), 22.0, S.mat(P.SLIME, glow=1.0))
    out += [lamp, S.turned((0, 0, 200.0), [(0, 0), (28.0, 0), (0.1, 26.0)], k.roof)]
    return Part("monument", boxes=out)
