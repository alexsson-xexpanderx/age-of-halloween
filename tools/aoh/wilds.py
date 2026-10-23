"""The map's scenery, haunted: trees, stumps, mines, bushes, rocks and plants, the map decorations (gaia) and every
projectile. Each takes what Age of Minecraft's took (props.py builds them), so the game's pictures are replaced one
for one; trees and mines take their size from the original picture they replace ("fit").

Trees by kind: oaks are twisted and bare or in sparse autumn leaves (orange, rust, blood red), birches pale and
ghostly, spruces black with frost on the snowy ones, palms dead with drooping fronds, jungle trees hung with moss
and glowing mushrooms, bamboo black, cherries blood-red, the dead ones bare.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from . import haunted as H
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part
from .graveyard import skull
from .haunted import box, hull

UP = 1.5 * math.cos(math.radians(30))  # screen pixels per model unit of height
AUTUMN = [P.hexrgb("#c8501a"), P.hexrgb("#8a2a14"), P.hexrgb("#d88a20"), P.hexrgb("#5a1a14")]


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(" ".join(str(k) for k in key).encode()))


def _leaves(colour, seed=0) -> S.Material:
    c = np.asarray(colour, float)

    def color(look):
        return P.mix(c, c * 0.45, 0.55 * P.noise(look.p, 1.2, seed))
    return S.Material(color=color, vivid=0.7)


def _branchy(height, rng, bark, lean=0.15, spread=0.7, depth_max=3, trunk_k=0.075) -> tuple[list, list]:
    """A crooked tree skeleton; returns (shapes, branch tips) for leaves to go on."""
    out, tips = [], []

    def branch(p, d, length, r, depth):
        pts = [p]
        for _ in range(4):
            d = d + rng.normal(0, 0.07 if depth == 0 else 0.3, 3)
            d[2] = max(d[2], 0.92 if depth == 0 else 0.1)  # the trunk stays upright, the branches claw outwards
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * length / 4)
        for k in range(4):
            ra, rb = r * (1 - 0.18 * k), r * (1 - 0.18 * (k + 1))
            out.append(S.rod(tuple(pts[k]), tuple(pts[k + 1]), ra, bark, rb))
            out.append(S.ball(tuple(pts[k + 1]), rb, bark))
        if depth >= depth_max or r < 0.35:
            tips.append(pts[-1])
            return
        for k in (2, 3, 4):
            if k < 4 and rng.random() < 0.2:
                continue
            nd = d + rng.normal(0, spread, 3)
            nd[2] = abs(nd[2]) * 0.8 + 0.45  # clawing upwards
            nd /= np.linalg.norm(nd)
            branch(pts[k], nd, length * rng.uniform(0.5, 0.68), r * (1 - 0.18 * k) * 0.72, depth + 1)

    r = height * trunk_k
    for a in np.linspace(0, 2 * math.pi, 5, endpoint=False) + rng.uniform(0, 1):  # roots
        out.append(S.rod((0, 0, r * 1.1), (math.cos(a) * r * 2.6, math.sin(a) * r * 2.6, -0.4), r * 0.65, bark,
                         r * 0.15))
    branch(np.zeros(3), np.array([rng.normal(0, lean), rng.normal(0, lean), 1.0]), height * 0.62, r, 0)
    return out, tips


def _spruce(height, rng, frost=False, seed=0) -> list:
    """A black spruce: tiers of drooping needle skirts round a thin trunk, frost on the tips if `frost`."""
    needles = S.Material(color=lambda l: P.mix(P.hexrgb("#1e2a24"), P.hexrgb("#08100c"), 0.6 * P.noise(l.p, 1.0, seed)),
                         vivid=0.3)
    snow = S.Material(color=lambda l: P.mix(P.hexrgb("#d8e4ec"), P.hexrgb("#8a98a8"), 0.4 * P.noise(l.p, 1.0, seed)))
    out = [S.rod((0, 0, 0), (0, 0, height), height * 0.035, P.bark(seed=seed), 0.3)]
    tiers = max(4, int(height / 14))
    for i in range(tiers):  # ragged drooping skirts: each tier a jagged ring of branch tips under a peak
        f = i / tiers
        z = height * (0.1 + 0.8 * f)
        r = height * 0.28 * (1 - f) + 2.5
        pts = [(0.0, 0.0, z + height * 0.2)]
        for a in np.linspace(0, 2 * math.pi, 11)[:-1] + rng.uniform(0, 0.6):
            rr = r * rng.uniform(0.7, 1.1)
            pts.append((math.cos(a) * rr, math.sin(a) * rr, z - rng.uniform(0, height * 0.04)))
            pts.append((math.cos(a + 0.3) * rr * 0.55, math.sin(a + 0.3) * rr * 0.55, z + height * 0.05))
        out.append(hull(pts, needles))
        if frost:
            out.append(hull([(x * 0.6, y * 0.6, zz + height * 0.04) for x, y, zz in pts[1::2]] + [pts[0]], snow))
    return out


def tree(kind: str, height_px: float = 110.0, seed: int = 0, part: str = "all") -> Part:
    """A tree about `height_px` screen pixels tall. `part`: all, trunk (lower half) or crown (upper half)."""
    rng = _rng("tree", kind, seed)
    h = max(30.0, height_px / UP * 0.92)
    out = []
    if kind in ("spruce", "snowy_spruce"):
        out = _spruce(h, rng, frost=kind == "snowy_spruce", seed=seed)
    elif kind == "bamboo":
        stalk = S.Material(color=lambda l: P.mix(P.hexrgb("#1a1a1e"), P.hexrgb("#3a3a40"),
                                                 (np.abs(np.sin(l.p[:, 2] * 0.5)) > 0.95).astype(float)), shine=0.5)
        for i in range(7):
            x, y = rng.normal(0, 4, 2)
            hh = h * rng.uniform(0.7, 1.0)
            out.append(S.rod((x, y, 0), (x + rng.normal(0, 2), y + rng.normal(0, 2), hh), 1.2, stalk, 0.8))
            for j in range(3):
                z = hh * rng.uniform(0.5, 0.95)
                out.append(S.Ellipsoid(_leaves(P.hexrgb("#2a3020"), seed), S._orient((x, y, z), (rng.uniform(-40, 40), 0,
                                                                                               rng.uniform(0, 180))),
                                       radii=(4.0, 1.2, 0.4)))
    elif kind == "palm":
        bark = P.bark(P.hexrgb("#5a4a3a"), P.hexrgb("#2a2018"), seed)
        top = np.array([rng.normal(0, 6), rng.normal(0, 6), h * 0.85])
        pts = [np.zeros(3), top * 0.35 + (0, 0, 0), top * 0.7, top]
        for a, b in zip(pts, pts[1:]):
            out.append(S.rod(tuple(a), tuple(b), h * 0.03, bark, h * 0.026))
        frond = _leaves(P.hexrgb("#5a4a2a"), seed)
        for i in range(8):
            ang = 2 * math.pi * i / 8 + rng.uniform(0, 0.3)
            mid = top + np.array([math.cos(ang) * h * 0.18, math.sin(ang) * h * 0.18, -h * 0.02])
            end = top + np.array([math.cos(ang) * h * 0.3, math.sin(ang) * h * 0.3, -h * 0.22])
            for a, b, r in ((top, mid, h * 0.035), (mid, end, h * 0.025)):
                out.append(S.Ellipsoid(frond, S.axis_frame(a, b) @ S.affine(None, (0, 0, np.linalg.norm(b - a) / 2)),
                                       radii=(r, r * 0.25, np.linalg.norm(b - a) / 2)))
    else:
        bark = P.bark(P.hexrgb("#d0ccc4"), P.hexrgb("#3a3638"), seed) if kind == "birch" else P.bark(seed=seed)
        trunk_k = 0.05 if kind == "birch" else 0.075
        shapes, tips = _branchy(h, rng, bark, lean=0.12 if kind != "acacia" else 0.3,
                                spread=0.7 if kind != "acacia" else 1.1, trunk_k=trunk_k)
        out += shapes
        leaves = None
        if kind in ("oak", "dark_oak", "azalea") and seed % 3 != 2:
            leaves = AUTUMN[seed % len(AUTUMN)]
        elif kind == "cherry":
            leaves = P.hexrgb("#8a1420")
        elif kind == "jungle":
            leaves = P.hexrgb("#2a3a20")
        elif kind == "acacia":
            leaves = P.hexrgb("#4a3a20")
        if leaves is not None:  # sparse, ragged clumps of leaves on the branch ends
            for t in tips:
                if rng.random() < 0.75:
                    for j in range(3):
                        r = h * rng.uniform(0.025, 0.045)
                        c = np.asarray(t) + rng.normal(0, h * 0.03, 3)
                        col = AUTUMN[(seed + j) % len(AUTUMN)] if leaves is AUTUMN[seed % len(AUTUMN)] else leaves
                        out.append(hull(c + rng.normal(0, r, (6, 3)) * np.array([1, 1, 0.7]), _leaves(col, seed + j)))
        if kind == "jungle":  # hanging moss and glowing mushrooms at the foot
            moss = _leaves(P.hexrgb("#4a5a3a"), seed)
            for t in tips[:6]:
                out.append(S.rod(tuple(t), tuple(t - (0, 0, h * 0.25)), 0.6, moss, 0.1))
            for i in range(3):
                a = rng.uniform(0, 2 * math.pi)
                c = (math.cos(a) * h * 0.1, math.sin(a) * h * 0.1, 0.0)
                cap = S.ball((c[0], c[1], 2.4), (1.6, 1.6, 0.9), S.mat(P.hexrgb("#7cf0c8"), glow=0.8))
                out += [S.rod(c, (c[0], c[1], 2.4), 0.4, S.mat(P.hexrgb("#c8d8c0"))), cap.clip((0, 0, -1), -2.4)]
        if kind in ("dead", "oak") and seed % 5 == 0:  # a noose or a crow on some
            tip = max(tips, key=lambda t: t[2]) if tips else np.array([0, 0, h])
            out += [S.ball(tuple(tip + (0, 0, 1.4)), (1.0, 1.6, 1.1), S.mat(P.hexrgb("#0e0c10"))),
                    S.ball(tuple(tip + (0, 1.2, 2.2)), 0.8, S.mat(P.hexrgb("#0e0c10")))]
    root = Part("tree", boxes=out)
    if part != "all":
        mid = h * 0.5
        keep = []
        for s in out:
            z = (s.xf @ np.array([0, 0, 0, 1.0]))[2] if not isinstance(s, S.Convex) else s.bound[:, 2].mean()
            if (z < mid) == (part == "trunk"):
                keep.append(s)
        root = Part("tree", boxes=keep)
    return root


def stump(kind: str = "oak", variant: int = 0, felled: bool = True) -> Part:
    """What is left of a felled tree: a broken stump, splinters, a couple of logs."""
    rng = _rng("stump", kind, variant)
    bark = P.bark(P.hexrgb("#d0ccc4"), P.hexrgb("#3a3638"), variant) if kind == "birch" else P.bark(seed=variant)
    rings = S.Material(color=lambda l: P.mix(P.hexrgb("#8a6a4a"), P.hexrgb("#3a2a1a"),
                                             (np.sin(np.hypot(l.p[:, 0], l.p[:, 1]) * 2.5) > 0.4).astype(float) * 0.6))
    r = 5.0
    out = [S.rod((0, 0, 0), (0, 0, 6.0), r, bark, r * 0.9), S.rod((0, 0, 6.0), (0, 0, 6.2), r * 0.9, rings)]
    for i in range(4):
        a = rng.uniform(0, 2 * math.pi)
        out.append(S.rod((math.cos(a) * r * 0.6, math.sin(a) * r * 0.6, 6.0),
                         (math.cos(a) * r * 0.7, math.sin(a) * r * 0.7, 6.0 + rng.uniform(1.5, 4.0)), 0.8, bark, 0.1))
    if felled:
        out.append(S.rod((6.0, -4.0, 2.4), (20.0, 6.0, 2.4), 2.4, bark))
    return Part("stump", boxes=out)


def ore_pile(kind: str, size_px: float = 90.0, variant: int = 0) -> Part:
    """A mine: for gold, a broken coffin spilling coins and a skull among gold-veined black rocks; for stone, a heap
    of grey rock and old gravestones."""
    rng = _rng("ore", kind, variant)
    half = max(14.0, size_px / 1.5 / 1.414 * 0.5)
    rock = P.mottled(P.hexrgb("#3a3640"), P.hexrgb("#141218"), 2.0, 0.6, variant) if kind == "gold" else \
        P.mottled(P.hexrgb("#7a7680"), P.hexrgb("#3a3640"), 2.0, 0.5, variant)
    out = []
    for i in range(int(half / 4) + 4):
        c = np.array([rng.uniform(-half, half) * 0.8, rng.uniform(-half, half) * 0.8, 0.0])
        r = rng.uniform(4.0, 8.0)
        out.append(hull(c + rng.normal(0, r * 0.55, (7, 3)) * np.array([1, 1, 0.7]) + (0, 0, r * 0.5), rock))
    if kind == "gold":
        gold = S.Material(color=lambda l: P.mix(P.GOLD, P.hexrgb("#7a5a14"), P.noise(l.p, 0.6, 3)), shine=0.9, gloss=30,
                          glow=lambda l: 0.25 * P.noise(l.p, 0.8, 4), vivid=1.0)
        for i in range(int(half / 3) + 3):
            c = np.array([rng.uniform(-half, half) * 0.7, rng.uniform(-half, half) * 0.7, rng.uniform(1, 5)])
            out.append(hull(c + rng.normal(0, 1.6, (5, 3)), gold))
        coffin = P.wood(P.hexrgb("#2a1e18"), P.hexrgb("#0e0a08"), along=0, seed=variant)
        out.append(hull([(x, y, z) for x, y in ((-6, -3), (6, -3), (7, 0), (6, 3), (-6, 3), (-7, 0)) for z in (0.0, 3.0)],
                        coffin))
        for i in range(6):
            out.append(S.rod((rng.uniform(-5, 5), rng.uniform(-2, 2), 3.0), (rng.uniform(-5, 5), rng.uniform(-2, 2), 3.3),
                             1.0, gold))
        out += skull((rng.uniform(-half / 2, half / 2), half * 0.6, 0.5), 1.8, variant)
    else:
        for i in range(2):
            out += H.gravestone(rng.uniform(-half / 2, half / 2), rng.uniform(-half / 2, half / 2), 0.0, 6.0,
                                variant + i, seed=variant + i)
    return Part("ore", boxes=out)


def berry_bush(variant: int = 0, berries: float = 1.0) -> Part:
    """A forage bush: a thorny black bramble heavy with dark berries, a small pumpkin vine at its foot."""
    rng = _rng("bush", variant)
    leaves = _leaves(P.hexrgb("#2a3a24"), variant)
    out = [S.ball((rng.normal(0, 2), rng.normal(0, 2), 5.0), (8.0, 7.0, 6.0), leaves),
           S.ball((4.0, -3.0, 4.0), (5.0, 5.0, 4.0), leaves)]
    berry = S.mat(P.hexrgb("#3a1048"), shine=0.6, gloss=30, vivid=0.8)
    for i in range(int(14 * berries)):
        a, b = rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 1.2)
        out.append(S.ball((math.cos(a) * 7.5, math.sin(a) * 6.5, 5.0 + math.sin(b) * 5.0), 0.9, berry))
    out += PK.lantern((-8.0, 7.0, 0.0), 2.0, seed=variant) if variant % 2 == 0 else []
    return Part("bush", boxes=out)


def boulder(variant: int = 0, size_px: float = 60.0, stone=None) -> Part:
    rng = _rng("rock", variant)
    r = max(6.0, size_px / 1.5 / 1.414 * 0.4)
    rock = P.mottled(P.hexrgb("#5a5660"), P.hexrgb("#1e1c22"), 2.0, 0.6, variant)
    out = [hull(rng.normal(0, r * 0.6, (9, 3)) * np.array([1.2, 1.0, 0.7]) + (0, 0, r * 0.5), rock)]
    if variant % 3 == 0:  # a face in the rock
        out += skull((r * 0.5, r * 0.4, r * 0.4), r * 0.25, variant)
    return Part("rock", boxes=out)


def plants(variant: int = 0, flowers: bool = False, count: int = 5) -> Part:
    """Dead grass and glowing mushrooms; with flowers, black roses and purple nightshade."""
    rng = _rng("plants", variant, flowers)
    out = []
    grass = _leaves(P.hexrgb("#4a4a30"), variant)
    for i in range(count + 2):
        x, y = rng.uniform(-12, 12, 2)
        for j in range(3):
            out.append(S.rod((x, y, 0.0), (x + rng.normal(0, 1.2), y + rng.normal(0, 1.2), rng.uniform(3, 6)), 0.3,
                             grass, 0.02))
        if flowers:
            col = [P.hexrgb("#1a0a14"), P.hexrgb("#6a2a8a"), P.hexrgb("#8a1420")][i % 3]
            out.append(S.ball((x, y, 6.0), 1.0, S.mat(col, vivid=0.8)))
        elif i % 2 == 0:
            cap = S.ball((x + 2, y, 2.0), (1.4, 1.4, 0.8), S.mat(P.hexrgb("#7cf0c8"), glow=0.7))
            out += [S.rod((x + 2, y, 0.0), (x + 2, y, 2.0), 0.3, S.mat(P.hexrgb("#c8d8c0"))), cap.clip((0, 0, -1), -2.0)]
    return Part("plants", boxes=out)


def cactus(variant: int = 0) -> Part:
    """A dead cactus, grey-green and drooping, a bleached cow skull at its foot."""
    rng = _rng("cactus", variant)
    m = S.Material(color=lambda l: P.mix(P.hexrgb("#4a5a40"), P.hexrgb("#1e2a1a"),
                                         (np.abs(np.sin(np.arctan2(l.p[:, 1], l.p[:, 0]) * 6)) > 0.8).astype(float) * 0.5))
    out = [S.rod((0, 0, 0), (0, 0, 22.0), 3.0, m, 2.6), S.ball((0, 0, 22.0), 2.6, m)]
    for sx in (-1, 1):
        z = rng.uniform(8, 14)
        out += [S.rod((0, 0, z), (sx * 6.0, 0, z + 2.0), 1.8, m), S.rod((sx * 6.0, 0, z + 2.0), (sx * 6.0, 0, z + 9.0), 1.8,
                                                                        m), S.ball((sx * 6.0, 0, z + 9.0), 1.8, m)]
    out.append(S.Ellipsoid(P.bone(variant), S._orient((5.0, 6.0, 1.2)), radii=(1.6, 2.6, 1.2)))
    return Part("cactus", boxes=out)


def haystack(variant: int = 0) -> Part:
    """A haystack, rotten, a pitchfork stuck in it and a scarecrow's pumpkin head on top."""
    hay = H.thatch(P.hexrgb("#7a6a40"), seed=variant)
    out = [S.rod((0, 0, 0), (0, 0, 14.0), 10.0, hay, 6.0), S.ball((0, 0, 14.0), (6.0, 6.0, 5.0), hay)]
    out += [S.rod((4.0, 2.0, 10.0), (12.0, 8.0, 22.0), 0.4, P.wood(P.hexrgb("#4a3020"), P.hexrgb("#1e120a"), along=2))]
    out += PK.lantern((0.0, 0.0, 18.0), 2.6, seed=variant)
    return Part("haystack", boxes=out)


# --------------------------------------------------------------------------- map decorations

def yurt(variant: int = 0) -> Part:
    """A witch's hut: a crooked little cabin (or a tent of black hides), a cauldron steaming green outside."""
    k = H.kit("G", 1)
    rng = _rng("yurt", variant)
    if variant % 2 == 0:
        out = H.walls(k, -16, -14, 14, 12, 1.0, 16, door=("+y", -2.0, 6.0, 11.0), every=12.0)
        out += H.hip(k, -16, 14, -14, 12, 17.0, 22.0, over=3.0)
    else:
        hide = S.Material(color=lambda l: P.mix(P.hexrgb("#3a2e28"), P.hexrgb("#141010"), P.noise(l.p, 2.0, variant)))
        out = [S.rod((0, 0, 0), (0, 0, 30.0), 16.0, hide, 0.5)]
        out += [S.rod((rng.normal(0, 1), rng.normal(0, 1), 26.0), (rng.normal(0, 3), rng.normal(0, 3), 36.0), 0.5,
                      k.trim) for _ in range(4)]
    out += H.cauldron(18.0, 14.0, 0.0, 3.0, seed=variant)
    return Part("hut", boxes=out)


def pavilion(variant: int = 0) -> Part:
    """A carnival tent of the dead: striped canvas in faded colours on poles, lanterns."""
    cols = [(P.hexrgb("#6a2a3a"), P.hexrgb("#2a2a2a")), (P.hexrgb("#4a2a6a"), P.hexrgb("#1e1e22")),
            (P.hexrgb("#6a4a1a"), P.hexrgb("#1e1a14"))][variant % 3]
    canvas = S.Material(color=lambda l: np.where(((np.arctan2(l.p[:, 1], l.p[:, 0]) * 8 / math.pi) % 2 < 1)[:, None],
                                                 cols[0], cols[1]))
    out = [S.rod((0, 0, 0), (0, 0, 18.0), 22.0, canvas, 20.0), S.turned((0, 0, 18.0), [(0, 0), (24.0, 0), (0.1, 18.0)],
                                                                      canvas)]
    out += H.lanterns_hanging([(24.0, 0.0, 16.0), (0.0, 24.0, 16.0)], H.kit("F", 3))
    return Part("tent", boxes=out)


def ruins(variant: int = 0) -> Part:
    """A ruined chapel's wall with an empty pointed window, fallen stones, ivy."""
    rng = _rng("ruins", variant)
    stone = P.bricks(P.hexrgb("#6e6a74"), P.hexrgb("#34323a"), 4.0, 2.4, moss=1.4, seed=variant)
    k = H.kit("W", 3)
    out = [box((-26, -4, 0), (26, 4, 34.0), H.facade(stone, [H.Opening("+y", -6.0, 6.0, 10.0, 30.0, "gothic")], k))]
    for i in range(6):
        c = np.array([rng.uniform(-30, 30), rng.uniform(6, 20), 0.0])
        out.append(hull(c + rng.normal(0, 2.5, (5, 3)) * np.array([1, 1, 0.5]) + (0, 0, 1.5), stone))
    return Part("ruins", boxes=out)


def stone_head(variant: int = 0) -> Part:
    """A giant stone skull half sunk in the ground, moss in its sockets."""
    m = P.mottled(P.hexrgb("#6e6a74"), P.hexrgb("#2a2830"), 2.0, 0.5, variant)
    from .undead import _skull_material
    return Part("head", boxes=[S.Ellipsoid(_skull_material((0.4, 0.8, 0.3), variant), S._orient((0, 0, 8.0)),
                                           radii=(12.0, 13.0, 12.0)).clip((0, 0, -1), 0.0)])


def statue() -> Part:
    """A weeping angel on a plinth, its stone wings spread."""
    stone = P.mottled(P.hexrgb("#8a8690"), P.hexrgb("#3e3a46"), 2.0, 0.5, 3)
    out = [box((-8, -8, 0), (8, 8, 8.0), stone), S.rod((0, 0, 8.0), (0, 0, 30.0), 4.0, stone, 2.8),
           S.ball((0, 1.0, 33.0), 3.0, stone)]
    for sx in (-1, 1):
        out.append(hull([(sx * 2, -2, 28.0), (sx * 2, -2, 20.0), (sx * 14, -4, 34.0), (sx * 10, -4, 16.0),
                         (sx * 2, -2.6, 24.0)], stone))
    return Part("statue", boxes=out)


def graves(variant: int = 0) -> Part:
    rng = _rng("graves", variant)
    out = []
    for i in range(3):
        x, y = rng.uniform(-14, 14, 2)
        out += H.gravestone(x, y, 0.0, rng.uniform(5, 8), variant + i, seed=variant + i)
        out.append(box((x - 2.4, y + 1.0, 0.0), (x + 2.4, y + 9.0, 0.8), P.mottled(P.hexrgb("#3a2e24"))))
    return Part("graves", boxes=out)


def heads(variant: int = 0) -> Part:
    """Skulls on pikes."""
    rng = _rng("heads", variant)
    out = []
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=2)
    for i in range(3):
        x, y = rng.uniform(-10, 10, 2)
        out.append(S.rod((x, y, 0.0), (x, y, 20.0 + i * 2), 0.6, wood))
        out += skull((x, y, 19.0 + i * 2), 2.2, variant + i)
    return Part("heads", boxes=out)


def signpost(angle: float = 0.0) -> Part:
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=2)
    board = P.wood(P.hexrgb("#5a4030"), P.hexrgb("#2a1a10"), along=0)
    return Part("sign", rot=(0, 0, angle), boxes=[S.rod((0, 0, 0), (0, 0, 20.0), 0.8, wood),
                                                  hull([(-8, -0.6, 16), (6, -0.6, 16), (8, 0, 18.5), (6, 0.6, 21), (-8, 0.6, 21),
                                                        (-8, -0.6, 21), (6, -0.6, 21), (-8, 0.6, 16), (6, 0.6, 16)], board),
                                                  *skull((0, 0, 20.4), 1.2, 1)])


def standing_torch(t: float = 0.0) -> Part:
    """A torch on an iron stand burning green, its flame flickering with `t`."""
    iron = P.metal(P.hexrgb("#2c2a32"), rust=0.4)
    k = 1.0 + 0.25 * math.sin(t * 2 * math.pi * 2)
    f = S.Ellipsoid(S.mat(P.SLIME, glow=1.0), S._orient((0, 0, 24.0 + k)), radii=(2.2, 2.2, 4.0 * k))
    f.casts = False
    return Part("torch", boxes=[S.rod((0, 0, 0), (0, 0, 20.0), 0.6, iron), S.rod((0, 0, 20.0), (0, 0, 21.5), 1.6,
                                                                                 iron, 2.2), f])


def banner_flag(variant: int = 0, t: float = 0.0) -> Part:
    """A tattered banner in the owner's colour, waving with `t`."""
    wave = math.sin(t * 2 * math.pi) * 1.5
    if variant in (3, 4):
        wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=2)
        out = [S.rod((x, 0, 0), (x, 0, 40.0), 0.8, wood) for x in (-12.0, 12.0)]
        out += [hull([(-11, -0.4 + wave, 38), (11, -0.4 + wave, 38), (-11, 0.4, 20), (11, 0.4, 20), (0, wave, 16),
                      (-11, 0.4 + wave, 38), (11, 0.4 + wave, 38)], P.team(0.6))]
        return Part("banner", boxes=out)
    return Part("banner", boxes=H.team_flag((0, 0, 0), 40.0, variant))


def relic(glint: float = 0.0) -> Part:
    """The relic: a cursed golden skull on a little bone stand, glowing green from its sockets."""
    gold = P.metal(P.GOLD, shine=0.9)
    from .undead import _skull_material
    sk = _skull_material((0.5 + 0.5 * glint, 1.0, 0.4), 7)
    sk_gold = S.Material(color=lambda l: P.mix(sk.color(l), P.GOLD, 0.7), glow=sk.glow, shine=0.9, gloss=30, vivid=1.0)
    return Part("relic", boxes=[S.rod((0, 0, 0), (0, 0, 3.0), 3.0, P.bone(1), 2.0),
                                S.Ellipsoid(sk_gold, S._orient((0, 0.3, 6.6)), radii=(3.0, 3.2, 3.0))])


def skeleton(angle: float = 0.0) -> Part:
    """A skeleton lying on the ground."""
    from .undead import _ribs_material
    b = P.bone(5)
    out = [S.ball((0, 0, 2.0), (3.4, 4.0, 2.0), _ribs_material(5)).clip((0, 0, -1), -0.5)]
    out += skull((0, 7.0, 0.0), 2.2, 4)
    for sx in (-1, 1):
        out += [S.rod((sx * 2.0, -4.0, 0.8), (sx * 3.0, -14.0, 0.8), 0.6, b), S.rod((sx * 3.5, 2.0, 0.8),
                                                                                   (sx * 8.0, -4.0, 0.8), 0.5, b)]
    return Part("skeleton", rot=(0, 0, angle), boxes=out)


def rug(variant: int = 0) -> Part:
    cols = [P.hexrgb("#5a1a2a"), P.hexrgb("#2a1a4a"), P.hexrgb("#4a3a1a")]
    c = cols[variant % 3]
    m = S.Material(color=lambda l: np.where((np.abs(np.abs(l.p[:, 0]) - 18) < 2)[:, None] |
                                            (np.abs(np.abs(l.p[:, 1]) - 18) < 2)[:, None], c * 0.5, c))
    return Part("rug", boxes=[box((-22, -22, 0.0), (22, 22, 0.6), m)])


def broken_cart() -> Part:
    """An overturned cart with a coffin sliding off it."""
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    from .engines import wheel
    out = [hull([(x, y, z) for x in (-8, 8) for y in (-14, 14) for z in (0, 3)], wood),
           hull([(x, y, z) for x, y in ((-3, -4), (3, -4), (4, 0), (3, 6), (-3, 6), (-4, 0)) for z in (3.0, 6.0)],
                P.wood(P.hexrgb("#2a1e18"), P.hexrgb("#0e0a08"), along=0))]
    w = wheel("w", (10.0, 8.0, 5.0), 5.0, 3)
    return Part("cart", boxes=out, children=[Part("tilt", rot=(0, 30, 0), children=[w])])


def crater(variant: int = 0) -> Part:
    m = P.mottled(P.hexrgb("#2a2420"), P.hexrgb("#0a0806"), 3.0, 0.6, variant)
    return Part("crater", boxes=[S.torus((0, 0, 0), 18.0, 4.0, m), S.ball((0, 0, 0.2), (14.0, 14.0, 0.4),
                                                                          S.mat(P.hexrgb("#141010")))])


def sea_rock(variant: int = 0) -> Part:
    rng = _rng("searock", variant)
    m = P.mottled(P.hexrgb("#3e3a44"), P.hexrgb("#141218"), 3.0, 0.6, variant)
    return Part("rock", boxes=[hull(rng.normal(0, 10, (9, 3)) * np.array([1.2, 1.0, 1.4]) + (0, 0, 8), m)])


# --------------------------------------------------------------------------- projectiles

def projectile(kind: str, t: float = 0.5, spin: bool = False) -> Part:
    """`t` runs over the sprite's frames: the pitch for arrows, the spin for thrown things. Arrows are bone-tipped
    with black fletching; burning ones flame green; bolts are sharpened bones; javelins (the old snowball slot, the
    imps' javelins) are bone spears; mangonel stones are flaming skulls; bombard shot is a green fireball."""
    bone = P.bone(3)
    black = S.mat(P.hexrgb("#141012"))
    if kind in ("arrow", "fire_arrow", "bolt", "stuck_arrow"):
        L = 14.0 * (1.7 if kind == "bolt" else 1.0)
        out = [S.rod((0, -L / 2, 0), (0, L / 2, 0), 0.35 * (1.6 if kind == "bolt" else 1), P.wood(
            P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1) if kind != "bolt" else bone),
            S.rod((0, L / 2 - 0.5, 0), (0, L / 2 + 2.6, 0), 0.9, bone, 0.05)]
        for a in range(3):
            ang = a * 2 * math.pi / 3
            out.append(hull([(0, -L / 2, 0), (0, -L / 2 + 3.4, 0), (math.cos(ang) * 1.6, -L / 2 + 0.4, math.sin(ang) * 1.6)],
                            black))
        if kind == "fire_arrow":
            f = S.Ellipsoid(S.mat(P.SLIME, glow=1.0), S._orient((0, L / 2 - 1.0, 0)), radii=(1.6, 3.0, 1.6))
            f.casts = False
            out.append(f)
        body = Part("arrow", boxes=out)
        if kind == "stuck_arrow":
            return Part("projectile", rot=(-55, 0, 0), offset=(0, 0, 5), children=[body])
        return Part("projectile", rot=(0.0 if not spin else 50.0 - 100.0 * t, 0, 0), children=[body])
    if kind in ("axe", "sword"):
        from .creatures import weapon
        return Part("projectile", rot=(-360.0 * t, 0, 0), children=[Part("w", boxes=weapon("axe" if kind == "axe" else
                                                                                         "sword"))])
    if kind == "snowball":
        from .creatures import weapon
        return Part("projectile", children=[Part("w", boxes=weapon("javelin"))])
    if kind in ("cobble", "stone", "magma"):
        sk = skull((0, 0, -2.0), 3.0, 1)
        f = S.Ellipsoid(S.mat(P.hexrgb("#ff8a20") if kind != "magma" else P.SLIME, glow=1.0), S._orient((0, -3.0, 1.0)),
                        radii=(3.0, 4.6, 3.0))
        f.casts = False
        f.ink = False
        return Part("projectile", rot=(360.0 * t, 90.0 * t, 0), children=[Part("skull", boxes=[f] + sk)])
    ball = S.ball((0, 0, 0), 3.4, S.mat(P.hexrgb("#2a2830"), shine=0.6))
    f = S.Ellipsoid(S.mat(P.SLIME, glow=1.0), S._orient((0, -3.0, 0)), radii=(2.6, 4.0, 2.6))
    f.casts = False
    return Part("projectile", rot=(360.0 * t, 0, 0), children=[Part("shot", boxes=[f, ball])])
