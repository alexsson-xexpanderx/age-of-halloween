"""The main menu's button objects, as models drawn by sculpt.py like the units: a jack-o'-lantern on a chain, a
skull shield on crossed scythes, a spell book on a lectern between candles, a witch's cauldron, a crystal ball, and
gravestones and pumpkins for the ground.

Each faces +y, the side nightscape.model shows; seen that way, +x is on the picture's left.
"""
from __future__ import annotations

import math

import numpy as np

from . import haunted as Hn
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part

OPEN = 16.0
IRON = P.metal(P.hexrgb("#2a2830"), rust=0.25, seed=4, shine=0.7)
OAK = P.wood(P.hexrgb("#4a3324"), P.hexrgb("#1e130c"), along=2, scale=0.5, seed=3)


def lantern() -> Part:
    """The Single Player button: the giant's mean pumpkin face, lit from inside, hanging from its stem."""
    upper, lower, _ = PK.pumpkin_head(PK.SCARIER, team_leaf=False)
    upper.rot = (OPEN, 0, 0)  # the mouth a little open, showing the teeth and the candle
    return Part("lantern").add(upper, lower)


def _heater(scale: float, y0: float, y1: float, bulge: float) -> np.ndarray:
    outline = [(8, 10), (8.4, 3), (7.2, -3), (4.6, -8), (0, -11.5)]
    outline = outline + [(-u, z) for u, z in reversed(outline[:-1])]
    pts = [(u * scale, y, z * scale) for u, z in outline for y in (y0, y1)]
    return np.array(pts + [(0, y1 + bulge, -0.5 * scale)])


def _scaled(m: S.Material, k: float) -> S.Material:
    """A material drawn for a small shape, on one k times bigger."""
    def small(l):
        return S.Look(l.p / k, l.n, l.nw, l.face)
    return S.Material(color=(lambda l: m.color(small(l))) if callable(m.color) else m.color,
                      glow=(lambda l: m.glow(small(l))) if callable(m.glow) else m.glow,
                      shine=m.shine, gloss=m.gloss, vivid=m.vivid)


def shield() -> list:
    """The Multiplayer button: a dark purple shield in an iron rim, a skull on it, on two crossed scythes."""
    field = S.Material(color=lambda l: P.mix(P.hexrgb("#6a2a82"), P.hexrgb("#24102e"),
                                             P.fbm(l.p, 2.5, 11) * 0.8 + np.clip(-l.p[:, 2] / 14, 0, 0.4)),
                       shine=0.25, vivid=0.6)
    rim = P.metal(P.hexrgb("#6a4a2a"), rust=0.2, seed=6, shine=0.9)
    out = [S.hull(_heater(1.12, -1.6, 0.9, 0.6), rim), S.hull(_heater(1.0, -0.6, 1.1, 1.6), field)]
    from .undead import _jaw_material, _skull_material
    k = 1.8  # the skull, big in the middle (its material's eyes are drawn for a head about 2 across)
    out += [S.Ellipsoid(_scaled(_skull_material(seed=5), k), S._orient((0, 2.6, 1.6)), radii=(2.0 * k, 1.8 * k, 2.1 * k)),
            S.Ellipsoid(_scaled(_jaw_material(5), k), S._orient((0, 2.6 + 0.9 * k, 1.6 - 2.0 * k)),
                        radii=(1.4 * k, 1.2 * k, 0.8 * k))]
    for side in (1, -1):  # the scythes, crossed behind it: snath from low on one side to high on the other
        a, b = np.array([10.5 * side, -2.2, -13.0]), np.array([-10.0 * side, -2.2, 13.0])
        out.append(S.rod(a, b, 0.55, OAK))
        out += S.limb(b, b + (0, 0, 0.4), 0.7, IRON)
        blade = P.metal(P.hexrgb("#8a8a96"), rust=0.35, seed=7, shine=0.9)
        pts = []
        for k in range(9):  # the blade sweeps out and down from the top, thinning to a point
            t = k / 8
            reach = 11.5 * t
            pts.append(b + np.array([-side * reach, 0, -0.6 - 6.0 * t ** 2]))
        for k, (p, q) in enumerate(zip(pts, pts[1:])):
            w = 2.4 * (1 - k / 9) + 0.25
            out.append(S.hull([p + (0, -0.25, 0), p + (0, 0.25, 0), q + (0, -0.25, 0), q + (0, 0.25, 0),
                               p + (0, 0, -w * 1.4), q + (0, 0, -w * 1.2)], blade))
    return out


def _pages(seed: int) -> S.Material:
    """Yellowed pages written in lines of a spidery hand, a red rune here and there."""
    def color(l):
        p = l.p
        base = P.mix(P.hexrgb("#e8d9a8"), P.hexrgb("#a88a52"), P.fbm(p, 2.0, seed) * 0.7)
        line = (np.abs(((p[:, 1] + 0.3) % 1.1) - 0.55) < 0.14) & (np.abs(p[:, 0]) < 3.0)
        word = P.noise(p * (1, 0.2, 1), 0.6, seed + 1) > 0.42
        ink = line & word & (l.n[:, 2] > 0.5)
        rune = (np.hypot(p[:, 0] - 1.0, p[:, 1] + 1.5) < 0.9) & (l.n[:, 2] > 0.5)
        c = np.where(ink[:, None], P.hexrgb("#2a1a12"), base)
        return np.where(rune[:, None], P.hexrgb("#b0201a"), c)
    return S.Material(color=color, vivid=0.4)


def lectern() -> list:
    """The History button: an old spell book open on a lectern, a candle on either side."""
    out = [S.block((-5, -4, 0), (5, 4, 1.6), OAK),
           S.turned((0, 0, 1.6), [(0, 0), (3.4, 0), (2.4, 1.4), (2.0, 6), (2.8, 8), (2.2, 11), (3.4, 12.4), (0, 12.4)],
                    OAK)]
    tilt = (-30, 0, 0)  # the desk slopes down toward the viewer
    out.append(S.block((-8.5, -5.5, -0.7), (8.5, 5.5, 0.7), OAK, rot=tilt, at=(0, 0, 15.2)))
    R = S._orient((0, 0, 15.2), tilt)

    def placed(lo, hi, m, turn=0.0):
        sh = S.block(lo, hi, m, rot=(0, turn, 0))
        sh.xf = R @ sh.xf
        return sh

    cover = S.mat(P.hexrgb("#4a1414"), shine=0.2, vivid=0.5)
    out.append(placed((-7.6, -4.8, 0.7), (7.6, 4.8, 1.2), cover))
    for side in (1, -1):
        out.append(placed((0.2 if side > 0 else -7.2, -4.4, 1.2), (7.2 if side > 0 else -0.2, 4.4, 2.2), _pages(side + 3),
                          turn=-6 * side))
    for x in (11.5, -11.5):  # the candles on iron stands
        out += [S.rod((x, 0, 0), (x, 0, 9), 0.5, IRON), S.turned((x, 0, 0), [(0, 0), (2.2, 0), (0.6, 1.2), (0, 1.2)], IRON),
                S.turned((x, 0, 9), [(0, 0), (1.8, 0.2), (1.9, 0.8), (0, 0.8)], IRON),
                S.rod((x, 0, 9.6), (x, 0, 14 + (x > 0) * 2.5), 1.0, S.mat(P.hexrgb("#e6dcc0"), vivid=0.5))]
        top = 14 + (x > 0) * 2.5
        flame = S.Ellipsoid(S.mat(P.hexrgb("#ffd46a"), glow=1.0), S._orient((x, 0, top + 1.2)), radii=(0.5, 0.5, 1.2))
        flame.casts = False
        flame.ink = False
        out.append(flame)
    return out


def book() -> list:
    """An open spell book lying flat (for the Learn to Play banner)."""
    cover = S.mat(P.hexrgb("#4a1414"), shine=0.2, vivid=0.5)
    out = [S.block((-7.6, -4.8, 0), (7.6, 4.8, 0.5), cover)]
    for side in (1, -1):
        out.append(S.block((0.2 if side > 0 else -7.2, -4.4, 0.5), (7.2 if side > 0 else -0.2, 4.4, 1.5),
                           _pages(side + 3), rot=(0, -6 * side, 0)))
    return out


def cauldron() -> list:
    """The Options button: a witch's cauldron of green brew over a fire, on three legs."""
    out = Hn.cauldron(0, 0, 0, r=10.0, seed=2)
    for k in range(5):  # logs under it
        a = k * 2 * math.pi / 5 + 0.3
        out.append(S.rod((math.cos(a) * 9, math.sin(a) * 9, 0.8), (math.cos(a) * 2, math.sin(a) * 2, 2.4), 1.0,
                         P.bark(seed=k)))
    return out


def crystal_ball() -> list:
    """The Zone button: a crystal ball full of swirling green and purple mist, on an iron claw stand."""
    def mist(l):
        p = l.p
        s = P.fbm(p + np.sin(p[:, 2:3] * 0.8) * 2.0, 3.0, 21)
        return P.mix(P.hexrgb("#5a2a9a"), P.hexrgb("#8aff9a"), np.clip((s - 0.35) * 2.2, 0, 1))
    glass = S.Material(color=mist, glow=lambda l: 0.55 + 0.35 * P.fbm(l.p, 2.0, 22), shine=1.0, gloss=60, vivid=1.0)
    out = [S.turned((0, 0, 0), [(0, 0), (5.5, 0), (5.5, 1.0), (3.6, 1.8), (2.4, 3.6), (3.8, 5.0), (0, 5.0)], IRON),
           S.ball((0, 0, 10.2), 5.6, glass)]
    for k in range(3):
        a = k * 2 * math.pi / 3 + math.pi / 2
        c, s = math.cos(a), math.sin(a)
        out += S.limb((c * 3.2, s * 3.2, 4.8), (c * 5.4, s * 5.4, 8.0), 0.55, IRON, 0.35)
    return out


def gravestone(kind: int, seed: int = 0) -> list:
    stone = P.mottled(P.hexrgb("#6a6674"), P.hexrgb("#2e2a36"), 1.5, 0.45, seed)
    moss = S.Material(color=lambda l: P.mix(P.hexrgb("#6a6674"), P.hexrgb("#2a3a20"),
                                            np.clip((P.fbm(l.p, 1.2, seed) - 0.55) * 4 - l.p[:, 2] * 0.08, 0, 1)))
    if kind == 0:  # a rounded slab, leaning
        return [S.block((-3, -0.8, 0), (3, 0.8, 6), moss), S.rod((0, -0.8, 6), (0, 0.8, 6), 3.0, moss)]
    if kind == 1:  # a cross
        return [S.block((-0.8, -0.8, 0), (0.8, 0.8, 11), moss), S.block((-3.2, -0.8, 6.5), (3.2, 0.8, 8.1), stone),
                S.block((-2.2, -1.6, 0), (2.2, 1.6, 1.4), stone)]
    return [S.block((-2.2, -1.2, 0), (2.2, 1.2, 8), moss),  # an obelisk on a plinth
            S.hull([(-1.6, -0.9, 8), (1.6, -0.9, 8), (-1.6, 0.9, 8), (1.6, 0.9, 8), (0, 0, 11.5)], stone),
            S.block((-3.2, -2, 0), (3.2, 2, 1.4), stone)]


def jack(seed: int = 0, radius: float = 3.0) -> list:
    return PK.lantern((0, 0, 0), radius, seed=seed)
