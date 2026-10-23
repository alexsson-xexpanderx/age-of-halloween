"""Ghost ships, on the "ship" rig's parts (animation.py: the "root" rocks, "sail" billows, "weapon" kicks when it
fires), at Age of Minecraft's sizes. Hulls of black, rotting planks, torn sails in the owner's colour with a skull on
them, green lanterns, skeleton crews.
"""
from __future__ import annotations

import math

import numpy as np

from . import creatures as C
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .engines import IRON, WOOD, WOOD_UP
from .geometry import Part
from .graveyard import skull
from .units import Unit

DECK = 6.0


def planks(seed=0) -> S.Material:
    """Ship's planks: dark rotten boards running fore and aft, slimy green near the waterline."""
    w = P.wood(P.hexrgb("#3a2c24"), P.hexrgb("#16100c"), along=1, planks=1.6, seed=seed)

    def color(look):
        c = w.color(look)
        slime = np.clip((1.5 - look.p[:, 2]) * 0.35 + (P.fbm(look.p, 2.0, seed) - 0.5), 0, 1)
        return P.mix(c, P.hexrgb("#2e4a2a"), slime * 0.7)
    return S.Material(color=color)


def hull(length, width, depth=DECK, seed=0, stern_h=0.0, bow_h=0.0) -> list:
    """A hull along +y: the lower half of a long egg, its top cut flat for the deck; a raised stern castle and bow
    if asked."""
    m = planks(seed)
    deck = P.wood(P.hexrgb("#4a3a2e"), P.hexrgb("#221812"), along=1, planks=1.4, seed=seed + 1)
    m.cut = deck
    out = [S.Ellipsoid(m, S._orient((0, 0, depth)), radii=(width / 2, length / 2, depth + 1.0)).clip((0, 0, 1), 0.0)]
    rail = WOOD(seed + 2)
    out.append(S.torus((0, 0, depth + 0.4), 1.0, 0.0001, rail))  # (placeholder: no rail ring on an ellipse)
    out.pop()
    for sx in (-1, 1):  # gunwales
        pts = [np.array([sx * width / 2 * math.sqrt(max(0.0, 1 - (y / (length / 2)) ** 2)) * 0.98, y, depth + 0.5])
               for y in np.linspace(-length / 2 * 0.95, length / 2 * 0.95, 9)]
        for a, b in zip(pts, pts[1:]):
            out.append(S.rod(tuple(a), tuple(b), 0.5, rail))
    if stern_h:
        out.append(S.block((-width * 0.38, -length / 2 * 0.95, depth), (width * 0.38, -length / 2 * 0.55, depth + stern_h),
                           planks(seed + 3)))
        out.append(S.block((-width * 0.42, -length / 2 * 0.98, depth + stern_h), (width * 0.42, -length / 2 * 0.5,
                                                                                  depth + stern_h + 0.8), WOOD(seed)))
        for x in (-width * 0.2, width * 0.2):  # lit stern windows
            out.append(S.block((x - 0.8, -length / 2 * 0.97, depth + stern_h * 0.4), (x + 0.8, -length / 2 * 0.93,
                                                                                       depth + stern_h * 0.8),
                               S.mat(P.SLIME, glow=0.85)))
    if bow_h:
        out.append(S.hull([(x, y, z) for x in (-width * 0.3, width * 0.3) for y in (length / 2 * 0.55, length / 2 * 0.8)
                           for z in (depth, depth + bow_h)] + [(0, length / 2 * 0.95, depth + bow_h)], planks(seed + 4)))
    return out


def lantern_post(at, height=5.0, seed=0) -> list:
    at = np.asarray(at, float)
    top = at + (0, 0, height)
    lamp = S.ball(tuple(top - (0, 0, 0.8)), (0.9, 0.9, 1.1), S.mat(P.SLIME, glow=1.0))
    lamp.casts = False
    return [S.rod(tuple(at), tuple(top), 0.25, IRON(seed)), lamp,
            S.ball(tuple(top), (1.0, 1.0, 0.3), IRON(seed))]


def sail(name, mast_at, height, width, seed=0, skull_mark=True, square=True) -> list:
    """A mast with a torn square sail in the owner's colour, a skull painted on it: parts `name` (the sail, billowing
    from its yard) and `name`_mast."""
    mast_at = np.asarray(mast_at, float)
    top = mast_at + (0, 0, height)
    mast = Part(f"{name}_mast", boxes=[S.rod(tuple(mast_at), tuple(top + (0, 0, 2.0)), 0.6, WOOD_UP(seed)),
                                       S.rod(tuple(top + (-width / 2 - 0.5, 0, 0)), tuple(top + (width / 2 + 0.5, 0, 0)),
                                             0.35, WOOD(seed))])
    z0 = top[2] - height * 0.75

    def pattern(look):
        p = look.p
        holes = P.noise(p * np.array([1.0, 1.0, 1.0]), 1.6, seed) > 0.7
        k = np.where(holes, 0.18, 0.62 + 0.15 * P.noise(p, 3.0, seed + 2))
        if skull_mark:
            u, v = p[:, 0] - mast_at[0], p[:, 2] - (z0 + height * 0.42)
            head = (u / 2.2) ** 2 + (v / 2.4) ** 2 < 1
            eyes = ((((u - 0.9) / 0.6) ** 2 + ((v - 0.2) / 0.7) ** 2) < 1) | ((((u + 0.9) / 0.6) ** 2 +
                                                                          ((v - 0.2) / 0.7) ** 2) < 1)
            bones = (np.abs(np.abs(u) - np.abs(v + 2.8) * 1.1) < 0.45) & (v < -1.6) & (v > -5.0)
            k = np.where(head | bones, 0.98, k)
            k = np.where(eyes, 0.05, k)
        return k
    cloth = P.team(0.6, pattern)
    pts = []
    for x in (-width / 2, width / 2):
        for z, bulge in ((top[2] - 0.3, 0.0), (z0, 0.0)):
            pts += [(mast_at[0] + x, mast_at[1] + 0.4 + bulge, z), (mast_at[0] + x, mast_at[1] + 0.8 + bulge, z)]
    pts += [(mast_at[0], mast_at[1] + 2.4, (top[2] + z0) / 2)]
    body = S.hull(pts, cloth)
    return [Part(name, pivot=tuple(top), boxes=[body]), mast]


def crew(at, seed=0, n=2, spread=6.0) -> list:
    """A few skeleton sailors' heads and shoulders above the gunwale."""
    out = []
    rng = np.random.default_rng(seed)
    for k in range(n):
        p = np.asarray(at, float) + (rng.uniform(-spread / 4, spread / 4), (k - (n - 1) / 2) * spread, 0)
        out += skull(tuple(p + (0, 0, 2.6)), 1.2, seed + k)
        out.append(S.ball(tuple(p + (0, 0, 1.6)), (1.4, 1.0, 1.4), C.team_cloth(0.4, seed)))
    return out


def figure(at, kind, seed=0) -> list:
    """A sailor at a ship's scale (ships are drawn 2.4 times bigger than people, so about 12 high): the hooded
    ferryman of the dead with his pole and lantern, or a zombie fisherman with his rod."""
    at = np.asarray(at, float)
    out = []
    if kind == "ferryman":
        robe = C.cloth(P.hexrgb("#1c181e"), seed=seed)
        out += [S.rod(tuple(at), tuple(at + (0, 0, 8.0)), 2.2, robe, 1.4),
                S.ball(tuple(at + (0, -0.2, 9.0)), (1.7, 1.8, 1.9), robe).clip((0, 1, 0), 0.9)]
        out += skull(tuple(at + (0, 0.3, 7.9)), 0.9, seed)
        out += [S.rod(tuple(at + (1.6, 0.6, 15.0)), tuple(at + (2.0, -1.0, -4.0)), 0.25, WOOD_UP(seed)),
                S.ball(tuple(at + (1.5, 0.8, 6.0)), 0.5, P.bone(seed))]
        lamp = S.ball(tuple(at + (-1.8, 1.2, 5.0)), (0.7, 0.7, 0.9), S.mat(P.SLIME, glow=1.0))
        lamp.casts = False
        out += [S.rod(tuple(at + (-1.6, 0.8, 7.0)), tuple(at + (-1.8, 1.2, 5.8)), 0.12, IRON(seed)), lamp]
    else:
        skin = C.SKIN["zombie"](seed)
        shirt = C.cloth(C.DRAB[seed % len(C.DRAB)], seed=seed)
        out += [S.rod(tuple(at), tuple(at + (0, 0, 4.5)), 1.6, shirt, 1.8),
                S.ball(tuple(at + (0, 0.2, 5.6)), (1.9, 1.3, 1.6), shirt),
                S.Ellipsoid(C.eyes_material(skin, [(-0.5, 0.2, 0.28, 0.22), (0.5, 0.2, 0.28, 0.22)],
                                            C.EYES["zombie"], 0.55), S._orient(at + (0, 0.3, 8.2)),
                            radii=(1.2, 1.3, 1.4)),
                *S.limb(tuple(at + (1.6, 0.2, 6.4)), tuple(at + (1.8, 2.0, 5.2)), 0.45, skin),
                S.rod(tuple(at + (1.8, 2.0, 5.2)), tuple(at + (2.6, 8.0, 10.0)), 0.18, WOOD(seed)),
                S.rod(tuple(at + (2.6, 8.0, 10.0)), tuple(at + (2.8, 9.5, 0.0)), 0.05, S.mat(P.hexrgb("#c8c0b0")))]
    return out


def _ship(key, name, replaces, root, attack="fire", scale=2.4, civ=None) -> Unit:
    return Unit(key, name, replaces, "ship", root, group="ship", attack=attack, civ=civ, scale=scale)


def _cannons(ys, x, z, seed, length=3.2) -> list:
    out = []
    for y in ys:
        for sx in (-1, 1):
            out.append(S.rod((sx * (x - 0.5), y, z), (sx * (x + length), y, z + 0.2), 0.55, IRON(seed)))
    return out


def ghost_galley(key, name, replaces, size=0, seed=0) -> Unit:
    L, W = (28.0, 13.0) if size == 0 else (32.0, 14.0) if size == 1 else (38.0, 15.5)
    parts = hull(L, W, seed=seed, stern_h=3.5 + size, bow_h=1.5)
    parts += lantern_post((0, -L / 2 * 0.75, DECK + 3.5 + size), 4.0, seed)
    parts += lantern_post((0, L / 2 * 0.85, DECK + 1.5), 3.5, seed + 1)
    parts += crew((0, 2.0, DECK), seed, 2 + size, 5.0)
    if size >= 1:
        for x in (-W * 0.42, W * 0.42):  # a few oars dipping in
            for y in np.linspace(-L * 0.25, L * 0.25, 3 + size):
                parts.append(S.rod((x, y, DECK + 0.6), (x * 1.6, y - 1.0, 0.0), 0.25, WOOD(seed)))
    weapon = Part("weapon", pivot=(0, L / 2 * 0.7, DECK + 2.0),
                  boxes=[S.rod((0, L / 2 * 0.6, DECK + 2.0), (0, L / 2 * 0.98, DECK + 2.6), 0.7, IRON(seed))] +
                  (_cannons(np.linspace(-L * 0.2, L * 0.2, 2 + size), W / 2 * 0.95, DECK - 0.3, seed) if size >= 2
                   else []))
    root = Part("root").add(Part("ship_bow", boxes=[]), weapon, *sail("sail", (0, 2.0, DECK), 18 + 2 * size, 13 + size,
                                                                      seed))
    if size >= 1:
        root.add(*sail("sail_2", (0, -L * 0.25, DECK), 14 + 2 * size, 10 + size, seed + 1, skull_mark=False))
    root.boxes = parts
    return _ship(key, name, replaces, root, scale=2.6 if size == 2 else 2.4)


def ferry(key="transport_ship") -> Unit:
    """The ferry of the dead: a long flat barge, a hooded ferryman with a pole at the stern, lanterns fore and aft."""
    seed = 110
    L, W = 32.0, 15.0
    parts = hull(L, W, depth=4.0, seed=seed)
    parts += lantern_post((0, L / 2 * 0.85, 4.5), 6.0, seed)
    parts += lantern_post((0, -L / 2 * 0.6, 4.5), 4.0, seed + 1)
    parts += figure((0, -L / 2 * 0.72, 4.0), "ferryman", seed)
    root = Part("root", boxes=parts).add(Part("sail", boxes=[]), Part("sail_mast", boxes=[]))
    return _ship(key, "Ferry of the Dead", "Transport Ship", root, attack="none")


def merchant(key="trade_cog") -> Unit:
    """A ghostly merchant cog piled with coffins and crates."""
    seed = 120
    L, W = 28.0, 13.0
    parts = hull(L, W, seed=seed, stern_h=4.0, bow_h=2.0)
    for k, y in enumerate((-2.0, 4.0)):
        parts.append(S.hull([(x, y + dy, z) for x, dy in ((-2.4, -3.0), (2.4, -3.0), (3.0, 0.0), (2.4, 3.0),
                                                          (-2.4, 3.0), (-3.0, 0.0)) for z in (DECK, DECK + 2.6)],
                            WOOD(seed + k)))
    parts.append(S.block((-4.6, -8.0, DECK), (-1.0, -4.6, DECK + 3.0), WOOD_UP(seed)))
    parts += lantern_post((3.5, -L / 2 * 0.7, DECK + 4.0), 4.0, seed)
    root = Part("root", boxes=parts).add(*sail("sail", (0, 1.0, DECK), 19, 13, seed))
    return _ship(key, "Ghost Cog", "Trade Cog", root, attack="none")


def fishing_boat(key="fishing_ship") -> Unit:
    """A rowing boat with a zombie fisherman and a net, a lantern on a pole."""
    seed = 130
    parts = hull(22.0, 11.0, depth=4.0, seed=seed)
    parts += lantern_post((0, 8.5, 4.0), 7.0, seed)
    crew_part = Part("crew", boxes=figure((0, -3.0, 4.0), "fisher", seed))
    net = S.Material(color=lambda l: np.where((np.sin(l.p[:, 0] * 4) * np.sin(l.p[:, 1] * 4) > 0.3)[:, None],
                                              P.hexrgb("#8a7a5a"), P.hexrgb("#3a3228")))
    parts.append(S.ball((3.0, 4.0, 4.6), (2.4, 2.0, 0.8), net))
    root = Part("root", boxes=parts).add(crew_part, Part("sail", boxes=[]), Part("sail_mast", boxes=[]))
    return _ship(key, "Zombie Fishing Boat", "Fishing Ship", root, attack="none")


def fire_ship(key, name, replaces, fast=False) -> Unit:
    """A galley with braziers of green fire fore and aft, spitting flame."""
    seed = 140 + fast
    L, W = 28.0, 13.0
    parts = hull(L, W, seed=seed, bow_h=1.5)
    for y in (-L / 2 * 0.6, L / 2 * 0.6):
        parts.append(S.rod((0, y, DECK), (0, y, DECK + 3.0), 1.6, IRON(seed), 2.4))
        f = S.Ellipsoid(S.mat(P.SLIME, glow=1.0), S._orient((0, y, DECK + 5.4)), radii=(1.8, 1.8, 3.0))
        f.casts = False
        parts.append(f)
    parts += crew((0, 0.0, DECK), seed, 2, 5.0)
    weapon = Part("weapon", pivot=(0, L / 2 * 0.7, DECK + 2.0), boxes=[
        S.rod((0, L / 2 * 0.65, DECK + 2.0), (0, L / 2 * 1.0, DECK + 2.6), 0.9, IRON(seed), 1.3)])
    root = Part("root", boxes=parts).add(weapon, *sail("sail", (0, 0.0, DECK), 17, 12, seed))
    return _ship(key, name, replaces, root)


def coffin_boat(key, name, replaces, heavy=False) -> Unit:
    """A floating coffin stacked with powder kegs, a lit fuse fizzing on top: the demolition ship."""
    seed = 150 + heavy
    L, W = 24.0 if not heavy else 26.0, 11.0
    parts = hull(L, W, depth=4.0, seed=seed)
    parts.append(S.hull([(x, y, z) for x, y in ((-3.6, -9.0), (3.6, -9.0), (4.6, 2.0), (3.6, 8.0), (-3.6, 8.0),
                                                (-4.6, 2.0)) for z in (4.0, 7.5)], WOOD(seed)))
    keg = P.wood(P.hexrgb("#5a3a22"), P.hexrgb("#2a1a10"), along=2, seed=seed)
    for k, (x, y) in enumerate(((-1.8, -3.0), (1.8, -3.0), (0.0, 1.5)) + (((-1.8, 5.0), (1.8, 5.0)) if heavy else ())):
        parts.append(S.rod((x, y, 7.5), (x, y, 10.5), 1.6, keg))
        parts.append(S.torus((x, y, 8.0), 1.6, 0.18, IRON(seed)))
        parts.append(S.torus((x, y, 10.0), 1.6, 0.18, IRON(seed)))
    spark = S.ball((0, 1.5, 12.6), 0.6, S.mat(P.hexrgb("#fff0a0"), glow=1.0))
    spark.casts = False
    weapon = Part("weapon", boxes=[S.rod((0, 1.5, 10.5), (0.4, 1.0, 12.4), 0.15, S.mat(P.hexrgb("#2a2420"))), spark])
    root = Part("root", boxes=parts).add(weapon, *sail("sail", (0, -6.0, 4.0), 11, 8, seed, skull_mark=False))
    return _ship(key, name, replaces, root, attack="explode", scale=2.2)


def cannon_galleon(key, name, replaces, elite=False) -> Unit:
    seed = 160 + elite
    L, W = 40.0, 14.0
    parts = hull(L, W, seed=seed, stern_h=5.0, bow_h=2.0)
    parts += _cannons(np.linspace(-L * 0.25, L * 0.25, 4), W / 2 * 0.98, DECK - 0.6, seed, 3.6)
    parts += lantern_post((0, -L / 2 * 0.75, DECK + 5.0), 4.0, seed)
    parts += crew((0, 4.0, DECK), seed, 3, 5.0)
    big = S.rod((0, L / 2 * 0.6, DECK + 2.2), (0, L / 2 * 1.05, DECK + 2.8), 1.1 if elite else 0.9, IRON(seed))
    weapon = Part("weapon", pivot=(0, L / 2 * 0.6, DECK + 2.2), boxes=[big])
    root = Part("root", boxes=parts).add(weapon, *sail("sail", (0, 4.0, DECK), 20, 14, seed),
                                         *sail("sail_2", (0, -L * 0.22, DECK), 16, 11, seed + 1, skull_mark=False))
    return _ship(key, name, replaces, root, scale=2.6)


def death_longboat(key="longboat") -> Unit:
    """A draugr longboat: a dragon-skull prow, a row of skull-painted shields, a striped sail."""
    seed = 170
    L, W = 40.0, 12.0
    parts = hull(L, W, depth=5.0, seed=seed)
    bone = P.bone(seed)
    neck = [np.array([0, L / 2 * 0.85, 5.0]), np.array([0, L / 2 * 0.98, 9.0]), np.array([0, L / 2 * 0.95, 13.0])]
    for a, b in zip(neck, neck[1:]):
        parts += S.limb(tuple(a), tuple(b), 1.0, bone, 0.8)
    parts.append(S.Ellipsoid(C.eyes_material(bone, [(-0.9, 0.5, 0.5, 0.4), (0.9, 0.5, 0.5, 0.4)], P.SLIME, 0.6),
                             S._orient(neck[-1] + (0, 1.6, 0.2)), radii=(1.5, 2.6, 1.4)))
    for sx in (-1, 1):
        for y in np.linspace(-L * 0.3, L * 0.3, 6):
            parts.append(S.rod((sx * W / 2 * 0.9, y, 5.6), (sx * W / 2 * 1.02, y, 5.6), 1.6, C.team_cloth(0.6, seed)))
    weapon = Part("weapon", pivot=(0, 0, 6.0), boxes=[])
    root = Part("root", boxes=parts).add(Part("ship_bow", boxes=[]), weapon, *sail("sail", (0, 0.0, 5.0), 18, 14, seed))
    return _ship(key, "Draugr Longboat", "Longboat", root, civ="Vikings")


def shell_ship(key="turtle_ship") -> Unit:
    """A turtle ship whose deck is the shell of a giant dead sea turtle, studded with bone spikes, its skull for a
    prow breathing green smoke."""
    seed = 180
    L, W = 38.0, 16.0
    parts = hull(L, W, depth=4.0, seed=seed)
    shell = S.Material(color=lambda l: P.mix(P.hexrgb("#3e4a3a"), P.hexrgb("#1a2018"),
                                             np.clip(np.abs(np.sin(l.p[:, 0] * 0.7) * np.sin(l.p[:, 1] * 0.5)) * 1.6,
                                                     0, 1)), shine=0.3)
    parts.append(S.ball((0, -1.0, 4.0), (W / 2 * 0.95, L / 2 * 0.85, 8.0), shell).clip((0, 0, -1), -4.0))
    bone = P.bone(seed)
    for k in range(10):
        a = 2 * math.pi * k / 10
        base = np.array([math.cos(a) * W * 0.3, math.sin(a) * L * 0.3 - 1.0, 9.0])
        parts.append(S.rod(tuple(base), tuple(base + (0, 0, 2.6)), 0.6, bone, 0.05))
    parts.append(S.Ellipsoid(C.eyes_material(bone, [(-1.2, 0.6, 0.6, 0.5), (1.2, 0.6, 0.6, 0.5)], P.SLIME, 0.6),
                             S._orient((0, L / 2 * 0.88, 6.0)), radii=(3.0, 3.4, 2.6)))
    weapon = Part("weapon", pivot=(0, L / 2 * 0.9, 6.0), boxes=_cannons(np.linspace(-L * 0.2, L * 0.2, 3), W / 2 * 0.95,
                                                                          5.0, seed))
    root = Part("root", boxes=parts).add(weapon)
    return _ship(key, "Shell Ship", "Turtle Ship", root, civ="Koreans", scale=2.6)


SHIPS = {
    "fishing_ship": fishing_boat,
    "transport_ship": ferry,
    "trade_cog": merchant,
    "galley": lambda: ghost_galley("galley", "Ghost Galley", "Galley", 0, 101),
    "war_galley": lambda: ghost_galley("war_galley", "Ghost War Galley", "War Galley", 1, 102),
    "galleon": lambda: ghost_galley("galleon", "Ghost Galleon", "Galleon", 2, 103),
    "fire_ship": lambda: fire_ship("fire_ship", "Hellfire Ship", "Fire Ship"),
    "fast_fire_ship": lambda: fire_ship("fast_fire_ship", "Fast Hellfire Ship", "Fast Fire Ship", True),
    "demolition_ship": lambda: coffin_boat("demolition_ship", "Coffin Bomb", "Demolition Ship"),
    "heavy_demolition_ship": lambda: coffin_boat("heavy_demolition_ship", "Heavy Coffin Bomb",
                                                 "Heavy Demolition Ship", True),
    "cannon_galleon": lambda: cannon_galleon("cannon_galleon", "Phantom Cannon Galleon", "Cannon Galleon"),
    "elite_cannon_galleon": lambda: cannon_galleon("elite_cannon_galleon", "Elite Phantom Cannon Galleon",
                                                   "Elite Cannon Galleon", True),
    "longboat": death_longboat,
    "turtle_ship": shell_ship,
}
