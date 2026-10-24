"""Graveyard walls and dead trees.

Walls are the game's wall pieces (fortifications.PIECES order: "x" and "y" run along a model axis, "post" stands
where walls meet or end, "h" and "v" run across the screen): a low wall of mossy stone blocks with an iron railing of
spear-topped bars on it, and square pillars capped with a skull.
"""
from __future__ import annotations

import math

import numpy as np

from . import paint as P
from . import sculpt as S
from .geometry import Part

TILE = 96 / (1.5 * 2 ** 0.5)


def _stone(seed=0):
    return P.bricks(P.hexrgb("#8a8790"), P.hexrgb("#55525e"), w=3.4, h=2.0, moss=1.0, seed=seed)


def railing(a, b, z0, height, spacing=2.2) -> list:
    """Iron bars from a to b (on the ground plane), standing on z0, each with a spear point, joined by two rails."""
    iron = P.metal(P.hexrgb("#2c2a32"), rust=0.5, seed=4)
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    d = (b - a) / max(length, 1e-9)
    n = max(2, int(length / spacing) + 1)
    out = []
    for k in range(n):
        p = a + d * (k * length / (n - 1))
        top = z0 + height + (0.8 if k % 2 else 0.0)
        out.append(S.rod((p[0], p[1], z0), (p[0], p[1], top), 0.28, iron))
        out.append(S.hull([(p[0] - 0.55, p[1], top), (p[0] + 0.55, p[1], top), (p[0], p[1] - 0.55, top),
                           (p[0], p[1] + 0.55, top), (p[0], p[1], top + 1.6)], iron))
    for z in (z0 + 0.8, z0 + height - 0.9):
        out.append(S.rod((a[0], a[1], z), (b[0], b[1], z), 0.3, iron))
    return out


def skull(at, r=1.6, seed=0, glow=None) -> list:
    from .undead import _jaw_material, _skull_material
    c = np.asarray(at, float)
    m = _skull_material(seed=seed) if glow is None else _skull_material(glow, seed)
    return [S.Ellipsoid(m, S._orient(c + (0, 0, r * 1.05)), radii=(r, r * 1.05, r * 0.95)),
            S.Ellipsoid(_jaw_material(seed), S._orient(c + (0, r * 0.5, r * 0.3)), radii=(r * 0.68, r * 0.6, r * 0.38))]


def wall_piece(piece: str, seed: int = 0, height: float = 10.0) -> Part:
    """One wall piece, centred on its tile (half a tile each way)."""
    h = TILE / 2
    stone = _stone(seed)
    cap = P.bricks(P.hexrgb("#9a96a0"), P.hexrgb("#6a6672"), w=6.0, h=10.0, moss=0.6, seed=seed + 1)
    out = []
    if piece == "post":
        out.append(S.block((-5, -5, 0), (5, 5, height + 9), stone))
        out.append(S.block((-6, -6, height + 9), (6, 6, height + 10.5), cap))
        out.append(S.hull([(-4.5, -4.5, height + 10.5), (4.5, -4.5, height + 10.5), (-4.5, 4.5, height + 10.5),
                           (4.5, 4.5, height + 10.5), (0, 0, height + 13)], cap))
        out += skull((0, 0, height + 12.4), 2.0, seed)
        return Part("post", boxes=out)
    if piece in ("x", "h"):
        a, b = (-h, 0.0), (h, 0.0)
    else:
        a, b = (0.0, -h), (0.0, h)
    if piece in ("h", "v"):  # across the screen: a tile's diagonal
        ang = math.radians(45 if piece == "h" else -45)
        L = h * math.sqrt(2)
        a, b = (-math.cos(ang) * L, -math.sin(ang) * L), (math.cos(ang) * L, math.sin(ang) * L)
    a3, b3 = np.array([*a, 0.0]), np.array([*b, 0.0])
    d = (b3 - a3) / np.linalg.norm(b3 - a3)
    side = np.array([-d[1], d[0], 0.0])
    thick = 3.2
    pts = []
    for p in (a3, b3):
        for s in (-thick, thick):
            for z in (0.0, height * 0.55):
                pts.append(p + side * s + (0, 0, z))
    out.append(S.hull(pts, stone))
    cap_pts = []
    for p in (a3, b3):
        for s in (-thick - 0.6, thick + 0.6):
            for z in (height * 0.55, height * 0.55 + 1.3):
                cap_pts.append(p + side * s + (0, 0, z))
    out.append(S.hull(cap_pts, cap))
    out += railing(a3[:2], b3[:2], height * 0.55 + 1.3, height * 0.75)
    return Part("wall", boxes=out)


# --------------------------------------------------------------------------- dead trees

def dead_tree(height: float = 70.0, variant: int = 0) -> Part:
    """A twisted dead tree: a gnarled trunk splitting into crooked, clawing branches, no leaves."""
    rng = np.random.default_rng(1000 + variant)
    m = P.bark(seed=variant)
    out = []

    def branch(p, d, length, r, depth):
        pts = [p]
        for k in range(4):
            d = d + rng.normal(0, 0.12 if depth == 0 else 0.28, 3)
            d[2] = max(d[2], 0.8 if depth == 0 else 0.15 if depth < 2 else -0.3)  # the trunk stays upright
            d /= np.linalg.norm(d)
            pts.append(pts[-1] + d * length / 4)
        for k in range(4):
            ra = r * (1 - 0.18 * k)
            rb = r * (1 - 0.18 * (k + 1))
            out.append(S.rod(pts[k], pts[k + 1], ra, m, rb))
            out.append(S.ball(pts[k + 1], rb, m))
        if depth >= 3 or r < 0.5:
            return
        for k in (2, 3, 4):
            if k < 4 and rng.random() < 0.35:
                continue
            nd = d + rng.normal(0, 0.7, 3)
            nd[2] = abs(nd[2]) * 0.6 + 0.2
            nd /= np.linalg.norm(nd)
            branch(pts[k], nd, length * rng.uniform(0.5, 0.7), r * (1 - 0.18 * k) * 0.72, depth + 1)

    trunk_r = height * 0.085
    # roots flaring at the base
    for a in np.linspace(0, 2 * math.pi, 5, endpoint=False) + rng.uniform(0, 1):
        tip = np.array([math.cos(a) * trunk_r * 2.6, math.sin(a) * trunk_r * 2.6, -0.5])
        out.append(S.rod((0, 0, trunk_r * 1.2), tip, trunk_r * 0.7, m, trunk_r * 0.2))
    branch(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.05, 1.0]), height * 0.55, trunk_r, 0)
    return Part("tree", boxes=out)
