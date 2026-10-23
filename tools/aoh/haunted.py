"""Haunted buildings: a kit per haunted culture and age (`kit`), and every building made from it.

The five building sets become five haunted cultures, and the Dark Age is one shared look (G):

    G  Dark Age (everyone)  crooked log cabins under rotting thatch, mossy stones, a lantern by the door
    W  West European        haunted Victorian: peeling clapboard, then half-timbering, then soot-black brick;
                            steep purple shingle gables with iron cresting, witch-hat turrets, pumpkins
    E  Central/North        Transylvanian gothic: dark stone, pointed windows lit blood-red, needle spires
    M  Middle Eastern       mummy tombs: sandstone with hieroglyph bands, flat roofs and domes, green tomb light
    F  East Asian           ghost lanterns: black timber and paper walls, curved tile roofs, red paper lanterns
    X  Meso-American        Day of the Dead: bright painted plaster, stepped terraces, marigolds, candles

Every building stands centred on its footprint (the sprite's hotspot): 1x1 tile 48 units, 2x2 80, 3x3 128, 4x4 176,
5x5 224, as Age of Minecraft's (structures.py). Model +x and +y are the two walls the camera sees, so doors, windows
and decorations go there. Windows and doors are painted on the walls by `facade` (cheaper than shapes, and as
sharp): glass glowing with each culture's light, frames, mullions and arches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part

FOOT = {1: 48.0, 2: 80.0, 3: 128.0, 4: 176.0, 5: 224.0}


# --------------------------------------------------------------------------- shapes in part coordinates

def box(lo, hi, m: S.Material) -> S.Convex:
    """A box whose material sees part coordinates (so painted windows and brick courses line up across shapes)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    planes = [(np.array(v, float), float(s)) for v, s in (((1, 0, 0), hi[0]), ((-1, 0, 0), -lo[0]),
                                                          ((0, 1, 0), hi[1]), ((0, -1, 0), -lo[1]),
                                                          ((0, 0, 1), hi[2]), ((0, 0, -1), -lo[2]))]
    bound = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return S.Convex(m, np.eye(4), planes=planes, bound=bound)


def hull(points, m) -> S.Convex:
    return S.convex_from_points(np.asarray(points, float), m)


# --------------------------------------------------------------------------- materials

def logs(base=P.hexrgb("#4a3a2e"), dark=P.hexrgb("#1e1612"), d=2.6, seed=0) -> S.Material:
    """A log wall: round logs stacked, dark in the joints, bark texture."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(look):
        p = look.p
        f = (p[:, 2] / d) % 1.0
        shade = np.sin(f * math.pi) ** 0.6
        c = P.mix(dark, base, 0.25 + 0.75 * shade)
        return P.mix(c, dark, 0.3 * P.fbm(p * np.array([0.3, 0.3, 1.0]), 1.5, seed))
    return S.Material(color=color)


def thatch(base=P.hexrgb("#6a5a38"), dark=P.hexrgb("#2a2216"), seed=0) -> S.Material:
    """Rotting thatch: straw streaks running down the slope, dark and mossy in patches."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(look):
        p = look.p
        streak = P.noise(p * np.array([2.5, 2.5, 0.3]), 1.0, seed)
        c = P.mix(dark, base, 0.3 + 0.6 * streak)
        rot = np.clip((P.fbm(p, 6.0, seed + 3) - 0.55) * 3, 0, 1)
        return P.mix(c, P.MOSS * 0.7, rot * 0.6)
    return S.Material(color=color)


def plaster(base, dark=None, timber=None, grid=(9.0, 8.0), seed=0) -> S.Material:
    """Grimy plaster; with `timber`, a half-timbered frame of dark beams (posts, rails and braces) over it."""
    base = np.asarray(base, float)
    dark = base * 0.6 if dark is None else np.asarray(dark, float)

    def color(look):
        p = look.p
        c = P.mix(base, dark, 0.15 + 0.5 * P.fbm(p, 3.0, seed) ** 1.5)
        streaks = np.clip((P.noise(p * np.array([3.0, 3.0, 0.25]), 1.0, seed + 1) - 0.6) * 2, 0, 1)
        c = P.mix(c, dark * 0.7, streaks * 0.5)  # damp running down
        if timber is not None:
            n = np.abs(look.n)
            u = np.where(n[:, 0] > n[:, 1], p[:, 1], p[:, 0])
            fu = (u / grid[0]) % 1.0
            fz = (p[:, 2] / grid[1]) % 1.0
            beam = (np.minimum(fu, 1 - fu) * grid[0] < 0.7) | (np.minimum(fz, 1 - fz) * grid[1] < 0.6)
            brace = np.abs(fu - fz) * grid[0] < 0.6
            c = np.where((beam | brace)[:, None], np.asarray(timber, float) * (0.8 + 0.3 * P.noise(p, 1.0, seed)
                                                                               )[:, None], c)
        return c
    return S.Material(color=color)


def glyphs(base, ink=P.hexrgb("#3a2614"), seed=0) -> S.Material:
    """A band of hieroglyphs: little figures and signs pressed into sandstone."""
    base = np.asarray(base, float)

    def color(look):
        p = look.p
        n = np.abs(look.n)
        u = np.where(n[:, 0] > n[:, 1], p[:, 1], p[:, 0])
        cell_u = np.floor(u / 2.2)
        fu, fz = (u / 2.2) % 1.0, (p[:, 2] / 2.6) % 1.0
        pick = P._hash3(cell_u.astype(np.int64), np.floor(p[:, 2] / 2.6).astype(np.int64), 3, seed)
        mark = np.where(pick < 0.33, (np.abs(fu - 0.5) < 0.12) & (fz > 0.2) & (fz < 0.85),
                        np.where(pick < 0.66, np.hypot(fu - 0.5, fz - 0.55) < 0.25,
                                 (np.abs(fz - 0.5) < 0.1) & (np.abs(fu - 0.5) < 0.35)))
        c = P.mix(base, base * 0.75, P.noise(p, 2.0, seed))
        return np.where(mark[:, None], np.asarray(ink, float), c)
    return S.Material(color=color, vivid=0.4)


def painted(colours, seed=0, skulls=True) -> S.Material:
    """Day of the Dead plaster: walls painted in bright bands, a row of little sugar skulls along the top band."""
    cols = [np.asarray(c, float) for c in colours]

    def color(look):
        p = look.p
        z = p[:, 2]
        band = np.clip((z // 9.0).astype(int), 0, len(cols) - 1)
        c = np.stack([cols[b] for b in range(len(cols))])[band]
        c = P.mix(c, c * 0.6, 0.35 * P.fbm(p, 3.0, seed))  # faded and chipped
        chip = np.clip((P.fbm(p, 1.6, seed + 4) - 0.68) * 5, 0, 1)
        c = P.mix(c, P.hexrgb("#8a7a64"), chip)
        if skulls:
            n = np.abs(look.n)
            u = np.where(n[:, 0] > n[:, 1], p[:, 1], p[:, 0])
            fu, fz = (u / 6.0) % 1.0, (z / 9.0) % 1.0
            head = (np.hypot((fu - 0.5) * 6.0, (fz - 0.55) * 9.0 * 0.6) < 1.6) & (band == 1)
            eye = (np.hypot((np.abs(fu - 0.5) * 6.0) - 0.6, (fz - 0.6) * 9.0 * 0.6) < 0.4) & head
            c = np.where(head[:, None], P.hexrgb("#ece4d0"), c)
            c = np.where(eye[:, None], P.hexrgb("#1a1218"), c)
        return c
    return S.Material(color=color, vivid=0.75)


def tiles_curved(base=P.hexrgb("#2e3434"), dark=P.hexrgb("#121616"), seed=0) -> S.Material:
    """East Asian roof tiles: rows of round tiles running down the slope, ridged."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(look):
        p = look.p
        n = np.abs(look.n)
        u = np.where(n[:, 0] > n[:, 1], p[:, 1], p[:, 0])
        ridge = np.abs(np.sin(u * 1.6))
        c = P.mix(dark, base, 0.3 + 0.7 * ridge)
        return P.mix(c, dark, 0.3 * P.fbm(p, 2.5, seed))
    return S.Material(color=color, shine=0.35, gloss=18)


# --------------------------------------------------------------------------- the kit

@dataclass
class Kit:
    key: str
    age: int
    wall: S.Material
    upper: S.Material
    base: S.Material
    roof: S.Material
    trim: S.Material
    glow: np.ndarray
    roof_kind: str  # gable, spire, flat, pagoda, thatch, terrace
    window: str  # gothic, round, rect, lattice
    door_m: S.Material = None
    accent: np.ndarray = field(default_factory=lambda: P.hexrgb("#e08a20"))
    crest: bool = False


def kit(key: str, age: int) -> Kit:
    """The haunted culture `key` in `age` (1 Dark, 2 Feudal, 3 Castle, 4 Imperial)."""
    a = max(1, min(4, age))
    if key == "G" or a == 1:
        return Kit("G", 1, logs(seed=1), logs(seed=2), P.bricks(P.hexrgb("#6e6a72"), P.hexrgb("#3e3a44"), 3.4, 2.4,
                                                              moss=1.0, seed=3),
                   thatch(seed=4), P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=2, seed=5),
                   P.hexrgb("#ffa040"), "thatch", "rect")
    stone = P.bricks(P.hexrgb("#7a7680"), P.hexrgb("#45424c"), 4.0, 2.4, moss=0.8, seed=11)
    dark_wood = P.wood(P.hexrgb("#3a2a22"), P.hexrgb("#160e0a"), along=2, seed=12)
    if key == "W":
        walls = {2: siding(P.hexrgb("#5e5868"), peel=0.9, seed=13),
                 3: plaster(P.hexrgb("#8a8478"), timber=P.hexrgb("#24180f"), seed=14),
                 4: P.bricks(P.hexrgb("#5a3632"), P.hexrgb("#24140f"), 2.6, 1.3, moss=0.4, seed=15)}
        return Kit("W", a, walls[a], walls[min(a + 1, 4)] if a < 4 else walls[3], stone,
                   P.shingles(P.hexrgb("#4a3560"), P.hexrgb("#1e1428"), seed=16, axes=(0, 2)), dark_wood,
                   P.hexrgb("#ffa040"), "gable", "gothic", crest=a >= 3)
    if key == "E":
        walls = {2: logs(P.hexrgb("#3e3434"), P.hexrgb("#141010"), seed=21),
                 3: P.bricks(P.hexrgb("#5a5660"), P.hexrgb("#2a2830"), 3.2, 2.0, moss=0.6, seed=22),
                 4: P.bricks(P.hexrgb("#3e3a44"), P.hexrgb("#16141a"), 3.6, 2.2, moss=0.3, seed=23)}
        return Kit("E", a, walls[a], walls[3] if a == 2 else walls[a], stone,
                   P.shingles(P.hexrgb("#323a48"), P.hexrgb("#10141c"), w=1.6, h=1.2, seed=24, axes=(0, 2)),
                   dark_wood, P.hexrgb("#ff3a20"), "spire", "gothic", crest=True)
    if key == "M":
        sand = {2: P.bricks(P.hexrgb("#9a8460"), P.hexrgb("#5a4a34"), 3.0, 1.8, seed=31),
                3: P.bricks(P.hexrgb("#b8a070"), P.hexrgb("#6e5c3e"), 5.0, 2.6, seed=32),
                4: P.bricks(P.hexrgb("#c8b890"), P.hexrgb("#7e6e4e"), 6.0, 3.0, seed=33)}
        return Kit("M", a, sand[a], glyphs(P.hexrgb("#b8a070"), seed=34), sand[min(a + 1, 4)],
                   sand[a], P.wood(P.hexrgb("#5a4028"), P.hexrgb("#2a1c10"), along=2, seed=35),
                   P.hexrgb("#7cff6a"), "flat", "round", accent=P.GOLD)
    if key == "F":
        paper = plaster(P.hexrgb("#c8c0b0"), P.hexrgb("#6e685c"), timber=P.hexrgb("#141010"), grid=(7.0, 9.0),
                        seed=41)
        wood = P.wood(P.hexrgb("#2a1e1a"), P.hexrgb("#0e0a08"), along=2, seed=42)
        return Kit("F", a, wood if a == 2 else paper, paper, P.bricks(P.hexrgb("#6a6670"), P.hexrgb("#34323a"), 5.0,
                                                                     2.6, moss=0.6, seed=43),
                   tiles_curved(seed=44), P.wood(P.hexrgb("#5a1a14"), P.hexrgb("#2a0a08"), along=2, seed=45),
                   P.hexrgb("#ffc890"), "pagoda", "lattice", accent=P.hexrgb("#c8281a"))
    # X
    colours = {2: [P.hexrgb("#a0784a"), P.hexrgb("#a0784a"), P.hexrgb("#8a6440")],
               3: [P.hexrgb("#2a8a8a"), P.hexrgb("#d84a7a"), P.hexrgb("#e8a020")],
               4: [P.hexrgb("#6a3a9a"), P.hexrgb("#e0407a"), P.hexrgb("#2aa0a0")]}
    return Kit("X", a, painted(colours[a], seed=51, skulls=a >= 3), painted(colours[a][::-1], seed=52),
               P.bricks(P.hexrgb("#7a7468"), P.hexrgb("#3e3a34"), 5.0, 3.0, moss=1.0, seed=53),
               thatch(P.hexrgb("#7a6a40"), seed=54) if a == 2 else P.bricks(P.hexrgb("#8a8274"), P.hexrgb("#4a4438"),
                                                                           5.0, 2.6, moss=0.8, seed=54),
               P.wood(P.hexrgb("#4a3020"), P.hexrgb("#1e120a"), along=2, seed=55), P.hexrgb("#ffb040"),
               "thatch" if a == 2 else "terrace", "round", accent=P.hexrgb("#f08a10"))


def siding(paint_rgb, wood_rgb=P.WOOD_DARK, board: float = 2.2, peel: float = 0.5, seed=0) -> S.Material:
    """Painted clapboard: boards shadowed under each other, the paint faded and peeling to the wood."""
    paint_rgb, wood_rgb = np.asarray(paint_rgb, float), np.asarray(wood_rgb, float)

    def color(look):
        p = look.p
        f = (p[:, 2] / board) % 1.0
        c = P.mix(paint_rgb * 0.62, paint_rgb, np.clip(f * 1.6, 0, 1))
        c = P.mix(c, paint_rgb * 0.8, 0.35 * P.fbm(p, 4.0, seed))
        flake = np.clip((P.fbm(p * np.array([1, 1, 2.5]), 3.0, seed + 7) - 0.62) * 6 * peel, 0, 1)
        return P.mix(c, wood_rgb, flake)
    return S.Material(color=color)


# --------------------------------------------------------------------------- windows and doors, painted

@dataclass
class Opening:
    face: str  # +x, +y, -x, -y
    u0: float
    u1: float
    z0: float
    z1: float
    kind: str = "gothic"  # gothic, round, rect, lattice, door


def _inside(u, v, o: Opening, grow=0.0):
    uc = (o.u0 + o.u1) / 2
    hw = (o.u1 - o.u0) / 2 + grow
    z0, z1 = o.z0 - grow, o.z1 + grow
    du = np.abs(u - uc)
    ok = (v >= z0) & (v <= z1) & (du <= hw)
    if o.kind in ("gothic", "round", "door"):
        spring = z1 - hw * (1.3 if o.kind == "gothic" else 1.0)
        t = np.clip((v - spring) / max(z1 - spring, 1e-6), 0, 1)
        if o.kind == "gothic":
            allowed = hw * (1 - t ** 1.4)
        else:
            allowed = hw * np.sqrt(np.clip(1 - t * t, 0, 1))
        ok &= (v < spring) | (du <= allowed)
    return ok


def facade(base: S.Material, openings: list, k: Kit) -> S.Material:
    """`base` with painted windows and doors: glass glowing in the culture's light behind dark frames and mullions."""
    glow_rgb = np.asarray(k.glow, float)
    frame_rgb = P.hexrgb("#1a120e")
    door_m = k.door_m or P.wood(P.hexrgb("#3a2618"), P.hexrgb("#140c08"), along=2, seed=7)

    def classify(look):
        p, n = look.p, look.n
        glass = np.zeros(len(p), bool)
        frame = np.zeros(len(p), bool)
        door = np.zeros(len(p), bool)
        bars = np.zeros(len(p), bool)
        for o in openings:
            axis = 0 if o.face[1] == "x" else 1
            sign = 1 if o.face[0] == "+" else -1
            on = n[:, axis] * sign > 0.7
            if not on.any():
                continue
            u = p[:, 1] if axis == 0 else p[:, 0]
            v = p[:, 2]
            inner = on & _inside(u, v, o)
            ring = on & _inside(u, v, o, 0.6) & ~inner
            if o.kind == "door":
                door |= inner
                frame |= ring
                continue
            glass |= inner
            frame |= ring
            uc, vm = (o.u0 + o.u1) / 2, o.z0 + (o.z1 - o.z0) * 0.45
            if o.kind == "lattice":
                bars |= inner & ((np.abs(((u - o.u0) % 1.7) - 0.85) > 0.65) | (np.abs(((v - o.z0) % 1.7) - 0.85) > 0.65))
            else:
                bars |= inner & ((np.abs(u - uc) < 0.3) | (np.abs(v - vm) < 0.3))
        return glass & ~bars, frame | bars, door

    def color(look):
        c = base.color(look) if callable(base.color) else np.broadcast_to(np.asarray(base.color, float),
                                                                          (len(look.p), 3)).copy()
        glass, frame, door = classify(look)
        flicker = 0.8 + 0.2 * P.noise(look.p, 2.0, 5)
        c = np.where(glass[:, None], glow_rgb * flicker[:, None], c)
        c = np.where(frame[:, None], frame_rgb, c)
        if door.any():
            dc = door_m.color(look)
            c = np.where(door[:, None], dc, c)
        return c

    def glow(look):
        return classify(look)[0].astype(float)
    return S.Material(color=color, glow=glow, bump=base.bump, shine=base.shine, gloss=base.gloss, vivid=base.vivid)


def wall_openings(k: Kit, x0, y0, x1, y1, z0, h, door: Optional[tuple] = None, every: float = 16.0,
                  sill: float = 0.32, top: float = 0.78, width: float = 5.5, faces=("+x", "+y")) -> list:
    """Windows along the camera-facing walls, evenly spaced, and an optional door (face, centre, width, height)."""
    out = []
    kind = k.window
    for face in faces:
        a, b = (y0, y1) if face == "+x" else (x0, x1)
        n = max(1, int((b - a) // every))
        for i in range(n):
            c = a + (b - a) * (i + 0.5) / n
            if door and door[0] == face and abs(c - door[1]) < door[2] / 2 + width:
                continue
            out.append(Opening(face, c - width / 2, c + width / 2, z0 + h * sill, z0 + h * top, kind))
    if door:
        face, c, w, dh = door
        out.append(Opening(face, c - w / 2, c + w / 2, z0, z0 + dh, "door"))
    return out


# --------------------------------------------------------------------------- walls, roofs, towers

def walls(k: Kit, x0, y0, x1, y1, z0, h, door=None, every=16.0, m: S.Material = None, plinth: float = 2.0,
          windows=True) -> list:
    m = m or k.wall
    ops = wall_openings(k, x0, y0, x1, y1, z0, h, door, every) if windows else (
        wall_openings(k, x0, y0, x1, y1, z0, h, door, 1e9)[-1:] if door else [])
    out = [box((x0, y0, z0), (x1, y1, z0 + h), facade(m, ops, k) if ops else m)]
    if plinth:
        out.append(box((x0 - 1.0, y0 - 1.0, z0 - plinth if z0 >= plinth else 0.0), (x1 + 1.0, y1 + 1.0, z0 + plinth),
                       k.base))
    return out


def gable(k: Kit, x0, x1, y0, y1, z, rise, axis="x", over=3.0, m=None, end_m=None, thick=1.4) -> list:
    """A gable roof whose ridge runs along `axis`: two shingled slopes over the gable ends; iron cresting if the
    kit has it."""
    m = m or k.roof
    end_m = end_m or k.upper
    out = []
    if axis == "x":
        ym = (y0 + y1) / 2
        ridge = z + rise
        for ye in (y1 + over, y0 - over):
            drop = rise * over / ((y1 - y0) / 2)
            pts = []
            for x in (x0 - over, x1 + over):
                pts += [(x, ym, ridge + thick), (x, ym, ridge), (x, ye, z - drop), (x, ye, z - drop + thick)]
            out.append(hull(pts, m))
        for x in (x0, x1):
            out.append(hull([(x - 0.3, y0, z), (x + 0.3, y0, z), (x - 0.3, y1, z), (x + 0.3, y1, z),
                             (x - 0.3, ym, ridge - 0.4), (x + 0.3, ym, ridge - 0.4)], end_m))
        ends = ((x0 - over, ym, ridge + thick), (x1 + over, ym, ridge + thick))
    else:
        xm = (x0 + x1) / 2
        ridge = z + rise
        for xe in (x1 + over, x0 - over):
            drop = rise * over / ((x1 - x0) / 2)
            pts = []
            for y in (y0 - over, y1 + over):
                pts += [(xm, y, ridge + thick), (xm, y, ridge), (xe, y, z - drop), (xe, y, z - drop + thick)]
            out.append(hull(pts, m))
        for y in (y0, y1):
            out.append(hull([(x0, y - 0.3, z), (x0, y + 0.3, z), (x1, y - 0.3, z), (x1, y + 0.3, z),
                             (xm, y - 0.3, ridge - 0.4), (xm, y + 0.3, ridge - 0.4)], end_m))
        ends = ((xm, y0 - over, ridge + thick), (xm, y1 + over, ridge + thick))
    if k.crest:
        out += cresting(ends[0], ends[1])
    return out


def cresting(a, b, every=4.0) -> list:
    """Iron cresting along a ridge: a rail with little spear points."""
    iron = P.metal(P.hexrgb("#2c2a32"), rust=0.4, seed=9)
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    out = [S.rod(tuple(a), tuple(b), 0.3, iron)]
    for i in range(1, int(L // every)):
        p = a + (b - a) * (i * every / L)
        hgt = 2.4 if i % 2 else 1.5
        out += [S.rod(tuple(p), tuple(p + (0, 0, hgt)), 0.18, iron),
                hull([p + (-0.45, 0, hgt), p + (0.45, 0, hgt), p + (0, -0.45, hgt), p + (0, 0.45, hgt),
                      p + (0, 0, hgt + 1.3)], iron)]
    return out


def hip(k: Kit, x0, x1, y0, y1, z, rise, over=3.0, m=None, ridge=0.0) -> list:
    """A hip roof (a pyramid, or a ridge `ridge` long along x)."""
    m = m or k.roof
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    pts = [(x, y, z) for x in (x0 - over, x1 + over) for y in (y0 - over, y1 + over)]
    pts += [(xm - ridge / 2, ym, z + rise), (xm + ridge / 2, ym, z + rise)]
    return [hull(pts, m)]


def spire(cx, cy, z, r, h, m, finial=True) -> list:
    """A needle spire: a cone with an iron finial (a cross of spikes)."""
    out = [S.rod((cx, cy, z), (cx, cy, z + h), r, m, 0.08)]
    if finial:
        iron = P.metal(P.hexrgb("#2c2a32"), rust=0.4, seed=9)
        out += [S.rod((cx, cy, z + h - 1.0), (cx, cy, z + h + 5.0), 0.25, iron),
                S.rod((cx - 1.4, cy, z + h + 3.0), (cx + 1.4, cy, z + h + 3.0), 0.2, iron)]
    return out


def dome(cx, cy, z, r, m, onion=False, finial=P.GOLD) -> list:
    if onion:
        prof = [(0, 0), (r * 0.95, 0), (r * 1.08, r * 0.45), (r * 0.9, r * 0.95), (r * 0.45, r * 1.35), (0.1, r * 1.75)]
        out = [S.turned((cx, cy, z), prof, m)]
        top = z + r * 1.75
    else:
        out = [S.ball((cx, cy, z), (r, r, r * 0.95), m).clip((0, 0, -1), 0.0)]
        top = z + r * 0.95
    out.append(S.rod((cx, cy, top - 0.5), (cx, cy, top + 3.0), 0.3, P.metal(finial, shine=0.8)))
    out.append(S.ball((cx, cy, top + 3.2), 0.7, P.metal(finial, shine=0.8)))
    return out


def flat_roof(k: Kit, x0, y0, x1, y1, z, parapet=3.0, merlon=3.0, m=None) -> list:
    """A flat roof behind a parapet with merlons (the tomb-like M style)."""
    m = m or k.upper
    out = [box((x0 - 0.8, y0 - 0.8, z), (x1 + 0.8, y1 + 0.8, z + 1.2), k.base)]
    if parapet:
        for (a0, a1, b0, b1) in ((x0 - 0.8, x1 + 0.8, y1 - 0.4, y1 + 0.8), (x0 - 0.8, x1 + 0.8, y0 - 0.8, y0 + 0.4)):
            out.append(box((a0, b0, z + 1.2), (a1, b1, z + 1.2 + parapet * 0.5), m))
            n = max(2, int((a1 - a0) // (merlon * 2)))
            for i in range(n):
                u = a0 + (a1 - a0) * (i + 0.25) / n
                out.append(box((u, b0, z + 1.2 + parapet * 0.5), (u + merlon, b1, z + 1.2 + parapet), m))
        for (b0, b1, a0, a1) in ((y0 - 0.8, y1 + 0.8, x1 - 0.4, x1 + 0.8), (y0 - 0.8, y1 + 0.8, x0 - 0.8, x0 + 0.4)):
            out.append(box((a0, b0, z + 1.2), (a1, b1, z + 1.2 + parapet * 0.5), m))
            n = max(2, int((b1 - b0) // (merlon * 2)))
            for i in range(n):
                u = b0 + (b1 - b0) * (i + 0.25) / n
                out.append(box((a0, u, z + 1.2 + parapet * 0.5), (a1, u + merlon, z + 1.2 + parapet), m))
    return out


def pagoda_roof(k: Kit, x0, x1, y0, y1, z, rise, over=6.0, m=None) -> list:
    """A curved East Asian roof: a wide, shallow skirt under a steeper top, its corners swept up in horns."""
    m = m or k.roof
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    lo = [(x, y, z) for x in (x0 - over, x1 + over) for y in (y0 - over, y1 + over)]
    mid = [(x, y, z + rise * 0.35) for x in (x0 + 2, x1 - 2) for y in (y0 + 2, y1 - 2)]
    out = [hull(lo + [(x, y, z + 1.2) for x, y, _ in lo] + mid, m)]
    top = [(x, y, z + rise * 0.35) for x in (x0 + 2, x1 - 2) for y in (y0 + 2, y1 - 2)]
    out.append(hull(top + [(xm - (x1 - x0) * 0.15, ym, z + rise), (xm + (x1 - x0) * 0.15, ym, z + rise)], m))
    horn_m = P.metal(P.hexrgb("#3a3430"), seed=3)
    for x in (x0 - over, x1 + over):
        for y in (y0 - over, y1 + over):
            out.append(S.rod((x, y, z + 0.6), (x + math.copysign(1.6, x - xm), y + math.copysign(1.6, y - ym), z + 4.0),
                             0.8, horn_m, 0.15))
    return out


def terrace(k: Kit, x0, y0, x1, y1, z, steps=2, step=4.0, inset=4.0) -> list:
    """Stepped terraces of stone, each step a band of painted plaster (Day of the Dead) and a row of candles."""
    out = []
    for i in range(steps):
        a = inset * i
        out.append(box((x0 + a, y0 + a, z + step * i), (x1 - a, y1 - a, z + step * (i + 1)), k.roof if i % 2 else k.upper))
    return out


def roof_for(k: Kit, x0, x1, y0, y1, z, axis="x", scale=1.0) -> tuple[list, float]:
    """The kit's own roof over a hall; returns (shapes, top height)."""
    w = min(x1 - x0, y1 - y0)
    if k.roof_kind == "gable":
        r = w * 0.75 * scale
        return gable(k, x0, x1, y0, y1, z, r, axis), z + r
    if k.roof_kind == "spire":
        r = w * 1.0 * scale
        out = gable(k, x0, x1, y0, y1, z, r, axis, over=2.0)
        return out, z + r
    if k.roof_kind == "flat":
        return flat_roof(k, x0, y0, x1, y1, z), z + 4.0
    if k.roof_kind == "pagoda":
        r = w * 0.45 * scale
        return pagoda_roof(k, x0, x1, y0, y1, z, r), z + r
    if k.roof_kind == "terrace":
        return terrace(k, x0, y0, x1, y1, z), z + 8.0
    r = w * 0.6 * scale  # thatch
    return hip(k, x0, x1, y0, y1, z, r, over=3.5, ridge=(x1 - x0) * 0.3 if axis == "x" else 0.0), z + r


def tower(k: Kit, cx, cy, r, z0, h, flag=True, seed=0) -> list:
    """The culture's tower: W a round turret under a bent witch's hat, E a square tower with a needle spire and
    pinnacles, M a minaret with a balcony and a dome, F a stack of little curved roofs, X a stepped shrine with a
    fire on top."""
    out = []
    glass = facade(k.wall, [Opening(f, -1.3, 1.3, z0 + h * 0.6, z0 + h * 0.85, k.window) for f in ("+x", "+y")], k)
    if k.key in ("W", "G"):
        out.append(S.rod((cx, cy, z0), (cx, cy, z0 + h), r, k.wall))
        prof = [(0, 0), (r + 2.2, 0), (r + 2.0, 0.8), (r * 0.9, r * 1.2), (r * 0.55, r * 2.6), (r * 0.25, r * 3.8),
                (0.12, r * 4.6)]
        out.append(S.turned((cx, cy, z0 + h), prof, k.roof, rot=(5, -7, 0)))
        tip = np.array([cx, cy, z0 + h]) + S.rotation(5, -7, 0) @ np.array([0, 0, r * 4.6])
        for ang in (30, 100):
            a = math.radians(ang)
            out.append(box((cx + math.cos(a) * r - 0.4 - 0.8, cy + math.sin(a) * r - 1.0, z0 + h * 0.55),
                           (cx + math.cos(a) * r + 0.4 - 0.8, cy + math.sin(a) * r + 1.0, z0 + h * 0.8),
                           S.mat(k.glow, glow=1.0)))
    elif k.key == "E":
        out += walls(k, cx - r, cy - r, cx + r, cy + r, z0, h, every=r * 2.2, plinth=0.0)
        out += spire(cx, cy, z0 + h, r * 1.15, r * 4.5, k.roof)
        for sx in (-1, 1):
            for sy in (-1, 1):
                out += spire(cx + sx * r, cy + sy * r, z0 + h, 1.2, r * 1.4, k.roof, finial=False)
        tip = np.array([cx, cy, z0 + h + r * 4.5 + 5])
    elif k.key == "M":
        out.append(S.rod((cx, cy, z0), (cx, cy, z0 + h), r, k.wall, r * 0.85))
        out.append(S.rod((cx, cy, z0 + h * 0.62), (cx, cy, z0 + h * 0.66), r * 1.35, k.base))
        out.append(S.rod((cx, cy, z0 + h), (cx, cy, z0 + h + r * 1.2), r * 0.6, k.upper))
        out += dome(cx, cy, z0 + h + r * 1.2, r * 0.75, k.roof, onion=k.age >= 3)
        out.append(S.ball((cx, cy, z0 + h * 0.75), (r * 0.5, r * 0.5, r * 0.5), S.mat(k.glow, glow=0.9)))
        tip = np.array([cx, cy, z0 + h + r * 3.0])
    elif k.key == "F":
        tiers = max(2, int(h // 10))
        hh = h / tiers
        for i in range(tiers):
            rr = r * (1 - 0.12 * i)
            out += walls(k, cx - rr, cy - rr, cx + rr, cy + rr, z0 + hh * i, hh, every=rr * 3, plinth=0.0,
                         windows=i == tiers - 1)
            out += pagoda_roof(k, cx - rr, cx + rr, cy - rr, cy + rr, z0 + hh * (i + 1), rr * 0.6, over=rr * 0.6)
        out += lanterns_hanging([(cx + r + 2.0, cy, z0 + hh * 0.9), (cx, cy + r + 2.0, z0 + hh * 0.9)], k)
        tip = np.array([cx, cy, z0 + h + r * 0.6])
        out.append(S.rod(tuple(tip - (0, 0, 1)), tuple(tip + (0, 0, 6)), 0.4, P.metal(P.GOLD)))
    else:  # X
        steps = 3
        for i in range(steps):
            rr = r * (1 - 0.22 * i)
            out.append(box((cx - rr, cy - rr, z0 + h / steps * i), (cx + rr, cy + rr, z0 + h / steps * (i + 1)),
                           k.wall if i % 2 == 0 else k.base))
        fire = S.Ellipsoid(S.mat(P.hexrgb("#ffb030"), glow=1.0), S._orient((cx, cy, z0 + h + 2.4)),
                           radii=(r * 0.4, r * 0.4, r * 0.8))
        fire.casts = False
        out += [S.rod((cx, cy, z0 + h), (cx, cy, z0 + h + 1.4), r * 0.55, k.base, r * 0.7), fire]
        tip = np.array([cx, cy, z0 + h + 2.0])
    if flag:
        out += team_flag(tip, 10.0, seed)
    return out


# --------------------------------------------------------------------------- decorations

def team_flag(base, height=10.0, seed=0) -> list:
    """A tattered pennant in the owner's colour on an iron pole."""
    base = np.asarray(base, float)
    iron = P.metal(P.hexrgb("#2c2a32"), rust=0.3)
    top = base + (0, 0, height)

    def torn(look):
        return np.where(np.sin(look.p[:, 1] * 2.8 + look.p[:, 2] * 1.9) > 0.85, 0.3, 0.68)
    cloth = hull([top - (0, 0, 0.3), top - (0, 0, 3.8), top + (0.4, -7.0, -1.6), top + (0.4, -6.0, -3.0),
                  top + (0.6, -3.5, -2.0), top + (0.2, -3.5, -2.0)], P.team(1.0, torn))
    return [S.rod(tuple(base), tuple(top), 0.25, iron), cloth]


def wall_banner(x, y, z, face="+y", w=4.0, h=10.0, seed=0) -> list:
    """A long banner hanging on a wall in the owner's colour, a white skull painted on it, the hem torn."""
    def pattern(look):
        p = look.p
        u = p[:, 0] - x if face == "+y" else p[:, 1] - y
        v = p[:, 2] - (z - h * 0.35)
        skull = (u / 1.4) ** 2 + (v / 1.5) ** 2 < 1
        eyes = (((np.abs(u) - 0.55) / 0.4) ** 2 + ((v - 0.2) / 0.45) ** 2) < 1
        k = np.where(skull, 1.4, 0.58)
        return np.where(eyes, 0.08, k)
    m = P.team(1.0, pattern)
    if face == "+y":
        pts = [(x - w / 2, y + 0.2, z), (x + w / 2, y + 0.2, z), (x - w / 2, y + 0.6, z), (x + w / 2, y + 0.6, z),
               (x - w / 2, y + 0.4, z - h), (x + w / 2, y + 0.4, z - h), (x, y + 0.4, z - h - 2.0)]
    else:
        pts = [(x + 0.2, y - w / 2, z), (x + 0.2, y + w / 2, z), (x + 0.6, y - w / 2, z), (x + 0.6, y + w / 2, z),
               (x + 0.4, y - w / 2, z - h), (x + 0.4, y + w / 2, z - h), (x + 0.4, y, z - h - 2.0)]
    return [hull(pts, m)]


def lanterns_hanging(points, k: Kit, red=True) -> list:
    """Paper lanterns hanging from eaves: red (or the culture's glow), lit from inside, with black caps."""
    out = []
    col = P.hexrgb("#d8281a") if red else k.glow
    for p in points:
        p = np.asarray(p, float)
        lam = S.Ellipsoid(S.Material(color=lambda l, c=col: np.broadcast_to(c, (len(l.p), 3)),
                                     glow=lambda l: 0.75 + 0.2 * np.cos(l.p[:, 2] * 4.0), vivid=1.0),
                          S._orient(p), radii=(1.3, 1.3, 1.7))
        lam.casts = False
        out += [lam, S.rod(tuple(p + (0, 0, 1.6)), tuple(p + (0, 0, 2.2)), 0.6, S.mat(P.hexrgb("#141010"))),
                S.rod(tuple(p + (0, 0, 2.2)), tuple(p + (0, 0, 4.0)), 0.08, S.mat(P.hexrgb("#141010")))]
    return out


def gravestone(x, y, z=0.0, h=5.0, kind=0, seed=0) -> list:
    stone = P.mottled(P.hexrgb("#7a7684"), P.hexrgb("#3e3a46"), 1.5, 0.4, seed)
    if kind % 3 == 0:  # a rounded slab
        return [box((x - 1.6, y - 0.5, z), (x + 1.6, y + 0.5, z + h - 1.6), stone),
                S.rod((x, y - 0.5, z + h - 1.6), (x, y + 0.5, z + h - 1.6), 1.6, stone)]
    if kind % 3 == 1:  # a cross
        return [box((x - 0.5, y - 0.5, z), (x + 0.5, y + 0.5, z + h + 1), stone),
                box((x - 1.8, y - 0.5, z + h - 1.6), (x + 1.8, y + 0.5, z + h - 0.6), stone)]
    return [box((x - 1.8, y - 0.5, z), (x + 1.8, y + 0.5, z + h * 0.7), stone),  # an obelisk
            hull([(x - 1.2, y - 0.4, z + h * 0.7), (x + 1.2, y - 0.4, z + h * 0.7), (x - 1.2, y + 0.4, z + h * 0.7),
                  (x + 1.2, y + 0.4, z + h * 0.7), (x, y, z + h * 1.4)], stone)]


def cauldron(x, y, z=0.0, r=3.0, seed=0) -> list:
    """A witch's cauldron on three legs, green brew bubbling over a fire."""
    iron = P.metal(P.hexrgb("#24222a"), seed=seed, shine=0.6)
    brew = S.Material(color=lambda l: P.mix(P.hexrgb("#7cff6a"), P.hexrgb("#2a9a3a"), P.noise(l.p, 0.8, seed)),
                      glow=0.85, vivid=1.0)
    out = [S.ball((x, y, z + r * 1.1), (r, r, r * 0.9), iron).clip((0, 0, 1), r * 0.55),
           S.torus((x, y, z + r * 1.1 + r * 0.55), r * 0.82, r * 0.12, iron),
           S.rod((x, y, z + r * 1.1 + r * 0.5), (x, y, z + r * 1.1 + r * 0.62), r * 0.8, brew)]
    for k in range(3):
        a = 2 * math.pi * k / 3
        out.append(S.rod((x + math.cos(a) * r * 0.6, y + math.sin(a) * r * 0.6, z + r * 0.6),
                         (x + math.cos(a) * r * 0.8, y + math.sin(a) * r * 0.8, z), 0.3, iron))
    for k in range(2):
        b = S.ball((x + (k - 0.5) * r * 0.6, y + 0.3, z + r * 1.1 + r * 0.75 + k * 0.6), r * 0.18, brew)
        b.casts = False
        out.append(b)
    fire = S.Ellipsoid(S.mat(P.hexrgb("#ff9a30"), glow=1.0), S._orient((x, y, z + r * 0.35)),
                       radii=(r * 0.5, r * 0.5, r * 0.4))
    fire.casts = False
    return out + [fire]


def candles(x, y, z=0.0, n=3, seed=0) -> list:
    rng = np.random.default_rng(seed)
    wax = S.mat(P.hexrgb("#e8e0c8"), vivid=0.5)
    out = []
    for k in range(n):
        dx, dy = rng.uniform(-1.5, 1.5, 2)
        hgt = rng.uniform(1.2, 2.6)
        out.append(S.rod((x + dx, y + dy, z), (x + dx, y + dy, z + hgt), 0.35, wax))
        f = S.Ellipsoid(S.mat(P.hexrgb("#ffd060"), glow=1.0), S._orient((x + dx, y + dy, z + hgt + 0.5)),
                        radii=(0.25, 0.25, 0.5))
        f.casts = False
        out.append(f)
    return out


def marigolds(x, y, z=0.0, n=5, seed=0) -> list:
    """A heap of marigold flowers (Day of the Dead)."""
    rng = np.random.default_rng(seed)
    cols = [P.hexrgb("#f08a10"), P.hexrgb("#f0b020"), P.hexrgb("#e05a10")]
    return [S.ball((x + rng.uniform(-2, 2), y + rng.uniform(-2, 2), z + 0.6 + rng.uniform(0, 0.6)),
                   rng.uniform(0.7, 1.1), S.mat(cols[k % 3], vivid=1.0)) for k in range(n)]


def urn(x, y, z=0.0, h=4.0, m=None) -> list:
    m = m or P.mottled(P.hexrgb("#8a6a40"), P.hexrgb("#3e2a18"), 1.0, 0.4)
    return [S.turned((x, y, z), [(0, 0), (h * 0.25, 0), (h * 0.4, h * 0.35), (h * 0.32, h * 0.75), (h * 0.18, h * 0.85),
                                 (h * 0.24, h), (0, h)], m)]


def sarcophagus(x, y, z=0.0, axis="y", L=9.0, seed=0) -> list:
    """A painted mummy case standing against a wall, its gold face looking out."""
    gold = P.metal(P.GOLD, seed=seed, shine=0.8)
    body = S.Material(color=lambda l: P.mix(P.hexrgb("#2a5a8a"), P.GOLD,
                                            (np.sin(l.p[:, 2] * 2.2) > 0.6).astype(float)), shine=0.5, vivid=0.7)
    out = [S.Ellipsoid(body, S._orient((x, y, z + L / 2)), radii=(2.0, 1.4, L / 2)),
           S.Ellipsoid(gold, S._orient((x, y + (0.9 if axis == "y" else 0), z + L * 0.82)), radii=(1.3, 0.9, 1.4))]
    return out


def pumpkins(x, y, z=0.0, n=2, seed=0) -> list:
    out = []
    rng = np.random.default_rng(seed)
    for k in range(n):
        out += PK.lantern((x + rng.uniform(-2, 2), y + rng.uniform(-1, 1), z), rng.uniform(1.8, 2.6), seed=seed + k)
    return out


def bats(points, seed=0) -> list:
    """Bats flitting round a roof: little black wing shapes (they cast no shadow)."""
    m = S.mat(P.hexrgb("#100c10"))
    out = []
    for p in points:
        p = np.asarray(p, float)
        b = hull([p + (-2.4, 0, 0.6), p + (2.4, 0, 0.6), p + (-1.0, 0.2, -0.4), p + (1.0, 0.2, -0.4), p + (0, -0.4, 0),
                  p + (0, 0.4, 0.2)], m)
        b.casts = False
        b.ink = False
        out.append(b)
    return out


def fence(a, b, height=7.0, k: Kit = None) -> list:
    from .graveyard import railing
    return railing(a, b, 0.0, height)


def door_light(x, y, z, k: Kit) -> list:
    """A lantern on a bracket by the door (an iron one, or a paper one in F)."""
    if k.key == "F":
        return lanterns_hanging([(x, y, z)], k)
    iron = P.metal(P.hexrgb("#2c2a32"), rust=0.3)
    lamp = S.ball((x, y, z), (0.9, 0.9, 1.2), S.mat(k.glow, glow=1.0))
    lamp.casts = False
    return [lamp, box((x - 1.0, y - 1.0, z + 1.2), (x + 1.0, y + 1.0, z + 1.6), iron),
            S.rod((x, y, z + 1.6), (x, y - 1.6, z + 2.4), 0.15, iron)]
