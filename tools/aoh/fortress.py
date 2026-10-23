"""Walls, gates, towers, the outpost and the castle, in the haunted cultures (haunted.py kits).

Wall graphics hold five pieces (fortifications.PIECES order): "x" along model x, "y" along model y, "post" a pillar
filling its tile (where walls meet or end), "h" and "v" across the tile's diagonals. A wall tile is 48 units (half
24), as Age of Minecraft's three-block cells, so straight pieces reach their tile's edges and diagonals its corners.

    palisade   black sharpened stakes lashed together, skulls stuck on some
    stone      a graveyard wall: mossy stone in the culture's colour, a coping, an iron railing of spears
    fortified  a tall dark rampart with merlons, iron spikes and a walkway; the culture's coping and lights
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from . import haunted as H
from . import paint as P
from . import sculpt as S
from .geometry import Part
from .graveyard import railing, skull
from .haunted import box, hull

HALF = 24.0
PIECES = ("x", "y", "post", "h", "v")
GATE_DIRS = {"A": "x", "B": "y", "C": "h", "D": "v"}


def _stone(style: str, kind: str, seed=0) -> S.Material:
    cols = {"W": ("#7a7684", "#3e3a46"), "E": ("#5a5662", "#24222a"), "M": ("#a89470", "#5a4a34"),
            "F": ("#6e6a72", "#34323a"), "X": ("#7a7468", "#3a3630"), "G": ("#6e6a72", "#3a3640")}[style]
    w, h = (3.4, 2.0) if kind == "stone" else (5.0, 3.0)
    return P.bricks(P.hexrgb(cols[0]), P.hexrgb(cols[1]), w, h, moss=0.6 if style != "M" else 0.0, seed=seed)


def _ends(piece: str):
    if piece == "x":
        return np.array([-HALF, 0.0]), np.array([HALF, 0.0])
    if piece == "y":
        return np.array([0.0, -HALF]), np.array([0.0, HALF])
    if piece == "h":  # along x = -y (screen "--")
        return np.array([-HALF, HALF]), np.array([HALF, -HALF])
    return np.array([-HALF, -HALF]), np.array([HALF, HALF])  # v: along x = y (screen "|")


def _slab(a, b, thick, z0, z1, m, cut=None) -> S.Convex:
    """A straight wall slab from a to b (ground points), `thick` through, from z0 to z1."""
    a3, b3 = np.array([*a, 0.0]), np.array([*b, 0.0])
    d = (b3 - a3) / np.linalg.norm(b3 - a3)
    side = np.array([-d[1], d[0], 0.0])
    pts = [p + side * s + (0, 0, z) for p in (a3, b3) for s in (-thick / 2, thick / 2) for z in (z0, z1)]
    return hull(pts, m)


def wall_piece(kind: str, style: str, piece: str, damage: int = 0, progress: float = 1.0) -> Part:
    style = style if style in "WEMFX" else "G"
    rng = np.random.default_rng(zlib.crc32(f"{kind} {style} {piece} {damage}".encode()))
    out = []
    height = {"palisade": 42.0, "stone": 30.0, "fortified": 58.0}[kind]
    built = height * max(0.15, progress) if progress < 1 else height
    wood = P.wood(P.hexrgb("#2a201a"), P.hexrgb("#0e0a08"), along=2, seed=3)
    if piece == "post":
        if kind == "palisade":
            for i in range(9):
                x, y = (i % 3 - 1) * 10.0, (i // 3 - 1) * 10.0
                hgt = built + (6.0 if i == 4 else 0.0) - (rng.uniform(4, 14) if damage and rng.random() < 0.3 * damage
                                                           else 0.0)
                out += [S.rod((x, y, 0.0), (x, y, hgt), 3.4, wood, 3.0)]
                if progress >= 1:
                    out.append(S.rod((x, y, hgt), (x, y, hgt + 6.0), 3.0, wood, 0.2))
            if progress >= 1 and not damage:
                out += skull((0.0, 0.0, built + 11.0), 2.6, 1)
            return Part("post", boxes=out)
        stone = _stone(style, kind)
        r = 20.0
        top = built + 12.0
        out.append(box((-r, -r, 0.0), (r, r, top), stone))
        if progress >= 1:
            k = H.kit(style, 3)
            out.append(box((-r - 1.5, -r - 1.5, top), (r + 1.5, r + 1.5, top + 2.5), k.base))
            if style == "F":
                out += H.pagoda_roof(k, -r, r, -r, r, top + 2.5, 12.0, over=4.0)
            elif style == "E":
                out += H.spire(0.0, 0.0, top + 2.5, r * 0.9, 34.0, k.roof)
            elif style == "W" or style == "G":
                out += [S.turned((0, 0, top + 2.5), [(0, 0), (r + 1.0, 0), (r * 0.6, 10.0), (r * 0.25, 22.0), (0.1, 30.0)],
                                 k.roof, rot=(4, -5, 0))]
            elif style == "M":
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        out.append(box((sx * r - 5 if sx > 0 else -r, sy * r - 5 if sy > 0 else -r, top + 2.5),
                                       (sx * r if sx > 0 else -r + 5, sy * r if sy > 0 else -r + 5, top + 8.0), k.base))
            else:
                out += [box((-r * 0.6, -r * 0.6, top + 2.5), (r * 0.6, r * 0.6, top + 6.0), k.base)]
            out += skull((r + 0.5, 0.0, top - 6.0), 2.2, 2) if kind == "fortified" else []
            glow = box((r - 0.2, -2.0, top - 18.0), (r + 0.4, 2.0, top - 10.0), S.mat(k.glow, glow=1.0))
            out.append(glow)
        return Part("post", boxes=out)

    a, b = _ends(piece)
    L = float(np.linalg.norm(b - a))
    d = (b - a) / L
    if kind == "palisade":
        n = max(3, int(L // 7.0))
        for i in range(n):
            p = a + d * (L * (i + 0.5) / n)
            hgt = built + rng.uniform(-3, 3)
            if damage and rng.random() < 0.3 * damage:
                hgt *= rng.uniform(0.3, 0.7)
            out.append(S.rod((p[0], p[1], 0.0), (p[0], p[1], hgt), 3.3, wood, 3.0))
            if progress >= 1:
                out.append(S.rod((p[0], p[1], hgt), (p[0], p[1], hgt + 6.0), 3.0, wood, 0.2))
            if progress >= 1 and i == n // 2 and not damage:
                out += skull((p[0], p[1], hgt + 4.0), 2.2, 3)
        rope = S.mat(P.hexrgb("#5a4a34"))
        for z in (8.0, built - 8.0):
            out.append(S.rod((a[0], a[1], z), (b[0], b[1], z), 0.5, rope))
        return Part("wall", boxes=out)
    stone = _stone(style, kind)
    thick = 10.0 if kind == "stone" else 16.0
    k = H.kit(style, 3)
    segs = 4
    for i in range(segs):
        p0 = a + (b - a) * i / segs
        p1 = a + (b - a) * (i + 1) / segs
        hgt = built
        if damage and rng.random() < 0.25 * damage:
            hgt = built * rng.uniform(0.35, 0.8)
        out.append(_slab(p0, p1, thick, 0.0, hgt, stone))
        if progress >= 1 and hgt >= built * 0.99:
            out.append(_slab(p0, p1, thick + 2.5, hgt, hgt + 2.4, k.base))
            if kind == "stone":
                if not damage or rng.random() > 0.4:
                    out += railing(p0, p1, hgt + 2.4, 16.0, spacing=3.0)
            else:
                mid = (p0 + p1) / 2
                side = np.array([-d[1], d[0]])
                for s in (-1, 1):
                    c = mid + side * s * (thick / 2 - 2.0)
                    out.append(box((c[0] - 3.5, c[1] - 3.5, hgt + 2.4), (c[0] + 3.5, c[1] + 3.5, hgt + 10.0), stone))
                    out.append(S.rod((c[0], c[1], hgt + 10.0), (c[0], c[1], hgt + 14.0), 0.6,
                                     P.metal(P.hexrgb("#2c2a32"), rust=0.4), 0.05))
    if progress < 1:  # scaffolding beside the rising wall
        side = np.array([-d[1], d[0]])
        for t in (0.15, 0.85):
            p = a + (b - a) * t + side * (thick / 2 + 3.0)
            out.append(S.rod((p[0], p[1], 0.0), (p[0], p[1], built + 10.0), 0.6, wood))
    if damage >= 2:
        for i in range(3):
            c = a + (b - a) * rng.uniform(0.1, 0.9) + np.array([-d[1], d[0]]) * rng.uniform(6, 14)
            out.append(hull(np.array([c[0], c[1], 0.0]) + rng.normal(0, 2.5, (5, 3)) * np.array([1, 1, 0.5]) +
                            (0, 0, 1.5), stone))
    return Part("wall", boxes=out)


# --------------------------------------------------------------------------- gates

def gate_section(style: str, age: int, direction: str, is_open: bool, half: float = None) -> Part:
    """The gate between its two towers: a wall pierced by a gothic arch, a spiked iron portcullis (raised when open),
    a skull keystone; `half` (blocks) from the centre to a tower centre."""
    style = style if style in "WEMFX" else "G"
    kind = "fortified" if age >= 3 else "stone"
    diagonal = direction in ("h", "v")
    if half is None:
        half = 4.24 * (2 ** 0.5 if diagonal else 1)
    L = max(48.0, (2 * half - 2) * 16.0)
    height = 64.0 if kind == "fortified" else 48.0
    stone = _stone(style, kind)
    k = H.kit(style, max(2, age))
    gw, gh = 22.0, 34.0
    out = []
    t = 16.0
    out.append(box((-L / 2, -t / 2, 0.0), (-gw / 2, t / 2, height), stone))
    out.append(box((gw / 2, -t / 2, 0.0), (L / 2, t / 2, height), stone))
    out.append(box((-gw / 2, -t / 2, gh), (gw / 2, t / 2, height), stone))
    out.append(hull([(x, y, z) for x in (-gw / 2, gw / 2) for y in (-t / 2, t / 2) for z in (gh - 6.0, gh)] +
                    [(0, y, gh + 2.0) for y in (-t / 2, t / 2)], stone))
    out.append(box((-L / 2 - 1.5, -t / 2 - 1.5, height), (L / 2 + 1.5, t / 2 + 1.5, height + 3.0), k.base))
    for i in range(int(L // 14)):
        x = -L / 2 + 4.0 + i * 14.0
        out.append(box((x, -t / 2 - 1.5, height + 3.0), (x + 7.0, t / 2 + 1.5, height + 10.0), stone))
    out += skull((0.0, t / 2 + 1.0, gh + 4.0), 2.4, 4)
    iron = P.metal(P.hexrgb("#24222a"), rust=0.5)
    z0 = gh - 8.0 if is_open else 0.0
    for i in range(6):
        x = -gw / 2 + 2.0 + i * (gw - 4.0) / 5
        out.append(S.rod((x, 1.0, z0), (x, 1.0, gh - 2.0 if not is_open else gh + 4.0), 0.7, iron))
        if not is_open:
            out.append(S.rod((x, 1.0, 0.0), (x, 1.0, -0.1), 0.7, iron, 0.05))
    for z in (z0 + 6.0, z0 + 16.0, z0 + 26.0):
        out.append(S.rod((-gw / 2, 1.0, z), (gw / 2, 1.0, z), 0.5, iron))
    out += H.wall_banner(-gw / 2 - 8.0, t / 2 + 0.2, height - 4.0, "+y", 5.0, 14.0)
    out += H.wall_banner(gw / 2 + 8.0, t / 2 + 0.2, height - 4.0, "+y", 5.0, 14.0)
    rz = {"x": 0.0, "y": 90.0, "h": -45.0, "v": 45.0}[direction]
    return Part("gate", rot=(0, 0, rz), children=[Part("gate_body", boxes=out)])


def gate_tower(style: str, age: int) -> Part:
    style = style if style in "WEMFX" else "G"
    k = H.kit(style, max(2, age))
    stone = _stone(style, "fortified")
    height = 76.0 if age >= 3 else 62.0
    out = [box((-20.0, -20.0, 0.0), (20.0, 20.0, height), H.facade(stone, [
        H.Opening("+x", -3.0, 3.0, height - 26.0, height - 12.0, k.window),
        H.Opening("+y", -3.0, 3.0, height - 26.0, height - 12.0, k.window)], k))]
    out += _crown(k, style, 20.0, height, stone)
    out += H.wall_banner(20.0, 0.0, height - 30.0, "+x", 6.0, 16.0)
    return Part("gate_tower", boxes=out)


def _crown(k, style, r, top, stone) -> list:
    """What tops a square tower in each culture."""
    out = [box((-r - 2.0, -r - 2.0, top), (r + 2.0, r + 2.0, top + 3.0), k.base)]
    if style in ("W", "G"):
        out.append(S.turned((0, 0, top + 3.0), [(0, 0), (r + 2.5, 0), (r * 0.65, r * 0.7), (r * 0.3, r * 1.5),
                                                 (0.1, r * 2.1)], k.roof, rot=(4, -6, 0)))
    elif style == "E":
        out += H.spire(0.0, 0.0, top + 3.0, r * 0.95, r * 2.6, k.roof)
        for sx in (-1, 1):
            for sy in (-1, 1):
                out += H.spire(sx * r, sy * r, top + 3.0, 3.0, r * 0.9, k.roof, finial=False)
    elif style == "M":
        for i in range(4):
            a = i * math.pi / 2
            for j in range(3):
                u = -r + 4 + j * (2 * r - 8) / 2
                x, y = (u, r - 3) if i == 0 else (r - 3, u) if i == 1 else (u, -r + 3) if i == 2 else (-r + 3, u)
                out.append(box((x - 3.0, y - 3.0, top + 3.0), (x + 3.0, y + 3.0, top + 10.0), stone))
        out += H.dome(0.0, 0.0, top + 3.0, r * 0.55, k.roof, onion=True)
    elif style == "F":
        out += H.pagoda_roof(k, -r, r, -r, r, top + 3.0, r * 0.8, over=6.0)
    else:
        out.append(box((-r * 0.6, -r * 0.6, top + 3.0), (r * 0.6, r * 0.6, top + 9.0), k.wall))
        f = S.Ellipsoid(S.mat(P.hexrgb("#ffb030"), glow=1.0), S._orient((0, 0, top + 13.0)), radii=(4.0, 4.0, 6.0))
        f.casts = False
        out.append(f)
    return out


# --------------------------------------------------------------------------- towers

def watch_tower(style: str, level: int) -> Part:
    """Watch tower (1), guard tower (2), keep (3) or bombard tower (4), on one tile: taller and stouter each level,
    the bombard tower with a gargoyle cannon."""
    style = style if style in "WEMFX" else "G"
    k = H.kit(style, min(4, level + 1))
    r = {1: 13.0, 2: 15.0, 3: 18.0, 4: 18.0}[level]
    height = {1: 70.0, 2: 84.0, 3: 98.0, 4: 90.0}[level]
    stone = _stone(style, "fortified", seed=level) if level >= 2 else P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"),
                                                                             along=2)
    out = []
    if level == 1:  # a timber watch tower on four legs, a hut on top
        for sx in (-1, 1):
            for sy in (-1, 1):
                out.append(S.rod((sx * r, sy * r, 0.0), (sx * r * 0.7, sy * r * 0.7, height - 18.0), 1.6, stone))
        out.append(box((-r, -r, height - 20.0), (r, r, height - 17.0), stone))
        out += H.walls(k, -r * 0.8, -r * 0.8, r * 0.8, r * 0.8, height - 17.0, 12.0, every=10.0, plinth=0.0)
        roof, _ = H.roof_for(k, -r * 0.8, r * 0.8, -r * 0.8, r * 0.8, height - 5.0)
        out += roof
    else:
        out.append(box((-r, -r, 0.0), (r, r, height), H.facade(stone, [
            H.Opening("+x", -3.0, 3.0, height - 28.0, height - 14.0, k.window),
            H.Opening("+y", -3.0, 3.0, height - 28.0, height - 14.0, k.window),
            H.Opening("+y", -2.0, 2.0, height * 0.35, height * 0.5, k.window)], k)))
        out += _crown(k, style, r, height, stone)
    if level == 4:
        iron = P.metal(P.hexrgb("#24222a"), shine=0.7)
        out += [S.rod((r - 2.0, r - 2.0, height - 16.0), (r + 8.0, r + 8.0, height - 18.0), 2.6, iron, 2.2),
                S.ball((r + 8.5, r + 8.5, height - 18.0), 3.0, P.mottled(P.hexrgb("#4a4650"))),
                S.ball((r + 10.0, r + 10.0, height - 18.0), 1.4, S.mat(P.SLIME, glow=1.0))]
    out += H.team_flag((0.0, 0.0, height + (40.0 if style == "E" else 26.0)), 12.0, level)
    out += H.wall_banner(r, 0.0, height * 0.7, "+x", 5.0, 14.0) if level >= 2 else []
    return Part("tower", boxes=out)


def outpost() -> Part:
    """The outpost: a tall black post with a crow's nest, a lantern glowing green and a raven on top."""
    wood = P.wood(P.hexrgb("#2a201a"), P.hexrgb("#0e0a08"), along=2)
    out = [S.rod((0, 0, 0), (0, 0, 60.0), 2.4, wood, 1.8)]
    for a in range(3):
        ang = 2 * math.pi * a / 3
        out.append(S.rod((math.cos(ang) * 8, math.sin(ang) * 8, 0.0), (0, 0, 18.0), 1.0, wood))
    out += [S.rod((0, 0, 50.0), (0, 0, 54.0), 7.0, wood, 7.5)]
    lamp = S.ball((5.0, 5.0, 44.0), (1.6, 1.6, 2.2), S.mat(P.SLIME, glow=1.0))
    lamp.casts = False
    out += [lamp, S.rod((2.0, 2.0, 50.0), (5.0, 5.0, 46.4), 0.3, P.metal())]
    raven = S.mat(P.hexrgb("#100c12"), shine=0.5)
    out += [S.ball((0, 0, 58.0), (1.6, 2.6, 1.8), raven), S.ball((0, 2.2, 60.0), 1.2, raven),
            S.rod((0, 3.2, 60.0), (0, 4.8, 59.6), 0.4, raven, 0.05)]
    out += H.team_flag((-5.0, -5.0, 54.0), 10.0, 1)
    return Part("outpost", boxes=out)


# --------------------------------------------------------------------------- the castle

def castle(style: str, age: int = 3) -> Part:
    """The castle (4x4): a keep inside curtain walls with four towers and a gatehouse. W a haunted castle under
    purple witch-hat roofs, E Dracula's castle bristling with needle spires and red windows, M a tomb fortress of
    sandstone with domes, F a black keep of stacked curved roofs, X a temple fortress on a stepped pyramid with a
    rack of skulls."""
    style = style if style in "WEMFX" else "G"
    k = H.kit(style, max(3, age))
    stone = _stone(style, "fortified", seed=7)
    out = []
    R = 70.0
    wh = 44.0  # curtain wall height
    t = 12.0
    for (a, b) in (((-R, -R), (R, -R)), ((R, -R), (R, R)), ((-R, R), (R, R)), ((-R, -R), (-R, R))):
        out.append(_slab(np.array(a), np.array(b), t, 0.0, wh, stone))
        av, bv = np.array(a, float), np.array(b, float)
        d = (bv - av) / np.linalg.norm(bv - av)
        for i in range(10):
            p = av + (bv - av) * (i + 0.5) / 10
            out.append(box((p[0] - 3.0, p[1] - 3.0, wh), (p[0] + 3.0, p[1] + 3.0, wh + 6.0), stone))
    if style == "X":  # a stepped pyramid inside, a temple on top, a skull rack before it
        for i in range(4):
            a = 10.0 * i
            out.append(box((-50 + a, -50 + a, 10.0 * i), (40 - a, 40 - a, 10.0 * (i + 1)), k.wall if i % 2 == 0 else
                           k.base))
        out += H.walls(k, -16, -16, 6, 6, 40.0, 22, door=("+y", -5.0, 7.0, 14.0), plinth=0.0)
        out += H.terrace(k, -16, -16, 6, 6, 62.0, steps=2)
        for i in range(5):  # tzompantli: a rack of skulls
            x = 10.0 + i * 8.0
            out += skull((x, 58.0, 4.0), 2.2, i) + skull((x, 58.0, 10.0), 2.2, i + 5)
        out += [S.rod((8.0, 58.0, 14.0), (46.0, 58.0, 14.0), 0.6, k.trim)]
        keep_top = 72.0
    elif style == "M":
        out += H.walls(k, -44, -44, 24, 24, 0.0, 70, door=("+y", -10.0, 10.0, 26.0), m=stone)
        out += [box((-44.5, -44.5, 50.0), (24.5, 24.5, 56.0), k.upper)]
        out += H.flat_roof(k, -44, -44, 24, 24, 70.0)
        out += H.dome(-10.0, -10.0, 72.0, 22.0, k.roof, onion=True)
        keep_top = 110.0
    elif style == "F":
        tiers = 4
        for i in range(tiers):
            a = 7.0 * i
            out += H.walls(k, -44 + a, -44 + a, 24 - a, 24 - a, 22.0 * i, 22.0, door=("+y", -10.0, 10.0, 16.0) if i == 0
                           else None, plinth=0.0, m=stone if i == 0 else k.wall)
            out += H.pagoda_roof(k, -44 + a, 24 - a, -44 + a, 24 - a, 22.0 * (i + 1), 10.0, over=8.0)
        keep_top = 22.0 * tiers + 12.0
    else:
        out += H.walls(k, -44, -44, 20, 20, 0.0, 80, door=("+y", -12.0, 10.0, 24.0), m=stone)
        out += [box((-46, -46, 80.0), (22, 22, 84.0), k.base)]
        if style == "E":
            out += H.spire(-12.0, -12.0, 84.0, 22.0, 70.0, k.roof)
            for sx, sy in ((-40.0, -40.0), (16.0, -40.0), (-40.0, 16.0), (16.0, 16.0)):
                out += H.spire(sx, sy, 84.0, 5.0, 30.0, k.roof, finial=False)
            out += H.bats([(20.0, 30.0, 150.0), (-30.0, 10.0, 170.0), (0.0, -20.0, 160.0), (30.0, -10.0, 140.0)])
            keep_top = 160.0
        else:
            out.append(S.turned((-12, -12, 84.0), [(0, 0), (40.0, 0), (26.0, 18.0), (12.0, 40.0), (0.1, 56.0)], k.roof,
                                rot=(3, -4, 0)))
            keep_top = 140.0
    for (x, y) in ((-R, -R), (R, -R), (-R, R), (R, R)):
        out.append(S.rod((x, y, 0.0), (x, y, 66.0), 16.0, stone) if style in ("W", "G", "E") else
                   box((x - 16, y - 16, 0.0), (x + 16, y + 16, 66.0), stone))
        if style in ("W", "G"):
            out.append(S.turned((x, y, 66.0), [(0, 0), (18.5, 0), (12.0, 14.0), (5.0, 30.0), (0.1, 40.0)], k.roof,
                                rot=(4, -6, 0)))
        elif style == "E":
            out += H.spire(x, y, 66.0, 15.0, 46.0, k.roof)
        elif style == "M":
            out += H.dome(x, y, 66.0, 12.0, k.roof, onion=True)
        elif style == "F":
            out += H.pagoda_roof(k, x - 16, x + 16, y - 16, y + 16, 66.0, 12.0, over=6.0)
        else:
            out.append(box((x - 10, y - 10, 66.0), (x + 10, y + 10, 72.0), k.wall))
            f = S.Ellipsoid(S.mat(P.hexrgb("#ffb030"), glow=1.0), S._orient((x, y, 76.0)), radii=(4.0, 4.0, 6.0))
            f.casts = False
            out.append(f)
        out.append(box((x + (13.0 if x > 0 else -16.5), y - 2.5, 40.0), (x + (16.5 if x > 0 else -13.0), y + 2.5, 52.0),
                       S.mat(k.glow, glow=1.0)) if x > 0 else box((x - 2.5, y + 13.0, 40.0), (x + 2.5, y + 16.5, 52.0),
                                                                    S.mat(k.glow, glow=1.0)))
    # the gatehouse on the +y wall
    out += [box((-14.0, R - 10.0, 0.0), (14.0, R + 10.0, 60.0), H.facade(stone, [
        H.Opening("+y", -7.0, 7.0, 0.0, 28.0, "door")], k))]
    out += H.wall_banner(R + 6.0, -30.0, 40.0, "+x", 8.0, 22.0) + H.wall_banner(R + 6.0, 30.0, 40.0, "+x", 8.0, 22.0)
    out += H.wall_banner(-30.0, R + 6.0, 40.0, "+y", 8.0, 22.0) + H.wall_banner(30.0, R + 6.0, 40.0, "+y", 8.0, 22.0)
    out += H.team_flag((-12.0, -12.0, keep_top), 18.0, 5)
    return Part("castle", boxes=out)
