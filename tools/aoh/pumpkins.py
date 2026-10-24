"""Jack-o'-lanterns: Pac-Man as a scary carved pumpkin, the giant Pac-Man as a bigger, rotting, vine-grown one, and
small lanterns for the buildings and fields.

Pac-Man keeps his parts and the "pacman" rig (animation.py): the pumpkin is cut in two at the mouth (`jaw_top`,
`jaw_bottom`, hinged at the cut), so chomping opens it. His mouth is cut low, under a carved nose, so he reads as a
jack-o'-lantern rather than a ball cut in half, and in it lies a brain, lit from inside (the player asked); the giant
keeps his cut through the middle and his candle. Jagged teeth line both halves and
interlock when the mouth shuts; the angry eyes are carved openings glowing with the same light. His stem carries a
leaf in his owner's colour. "pop" is his death: the pumpkin bursts into glowing shards.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import paint as P
from . import sculpt as S
from .geometry import Part
from .units import Unit

R = 10.0  # Pac-Man's radius (as Age of Minecraft's: the game's room for him is set by it)
FLESH = P.hexrgb("#f7c77e")  # the cut pumpkin flesh


def _in_triangle(u, v, tri):
    (x1, y1), (x2, y2), (x3, y3) = tri
    d1 = (u - x2) * (y1 - y2) - (x1 - x2) * (v - y2)
    d2 = (u - x3) * (y2 - y3) - (x2 - x3) * (v - y3)
    d3 = (u - x1) * (y3 - y1) - (x3 - x1) * (v - y1)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _grow(tri, k):
    c = np.mean(tri, axis=0)
    return [tuple(c + (np.asarray(p) - c) * k) for p in tri]


def carved(features: list) -> callable:
    """A carve(look) for a pumpkin face: `features` are triangles in (azimuth, elevation) degrees as seen from the
    pumpkin's centre (azimuth 0 = its front, +y; positive to its left as we look at it, i.e. +x). Returns 1 inside
    an opening (candle light), 0 elsewhere; a thin band round each opening is left as darker cut edge by the colour."""
    def carve(look):
        p = look.p
        az = np.degrees(np.arctan2(p[:, 0], p[:, 1]))
        el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
        out = np.zeros(len(p))
        for tri in features:
            out = np.maximum(out, _in_triangle(az, el, tri).astype(float))
        return out
    return carve


def carved_edge(features: list) -> callable:
    def edge(look):
        p = look.p
        az = np.degrees(np.arctan2(p[:, 0], p[:, 1]))
        el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
        out = np.zeros(len(p), bool)
        for tri in features:
            out |= _in_triangle(az, el, _grow(tri, 1.22)) & ~_in_triangle(az, el, tri)
        return out
    return edge


def face_material(features, base=P.PUMPKIN, dark=P.PUMPKIN_DARK, rot=0.0, seed=0, glow=P.CANDLE) -> S.Material:
    """Pumpkin skin with carved, glowing openings and a dark cut edge round each."""
    carve, edge = carved(features), carved_edge(features)
    skin = P.pumpkin(base=base, dark=dark, rot=rot, seed=seed)
    glow = np.asarray(glow, float)

    def color(look):
        c = skin.color(look)
        e = edge(look)
        c = np.where(e[:, None], c * 0.32, c)
        k = carve(look)
        return P.mix(c, glow, k)

    return S.Material(color=color, glow=carve, bump=skin.bump, shine=skin.shine, gloss=skin.gloss, vivid=skin.vivid)


def inside(radius: float, glow=P.CANDLE) -> S.Material:
    """The cut face inside the mouth: candle light, brightest in the middle, with a rim of pale flesh."""
    glow = np.asarray(glow, float)

    deep = P.hexrgb("#3a0e04")

    def color(look):
        # a hollow lit by a candle: bright round the flame, falling off to a deep, dark red at the walls
        r = np.hypot(look.p[:, 0], look.p[:, 1] + radius * 0.1) / radius
        lit = P.mix(glow * 1.08, P.EMBER, np.clip(r / 0.45, 0, 1))
        lit = P.mix(lit, deep, np.clip((r - 0.4) / 0.45, 0, 1) ** 1.3)
        return np.where((r > 0.87)[:, None], FLESH * 0.8, np.clip(lit, 0, 1))

    def shine(look):
        r = np.hypot(look.p[:, 0], look.p[:, 1] + radius * 0.1) / radius
        return np.where(r > 0.87, 0.1, 1.0)

    return S.Material(color=color, glow=shine)


def _teeth(radius: float, z_sign: int, count: int, length: float, arc=(-75, 75), offset=0.0, seed=0,
           crooked=0.0) -> list:
    """Jagged teeth round the front of the cut, hanging down (z_sign -1) from the top jaw or up from the bottom one."""
    out = []
    rng = np.random.default_rng(seed)
    tooth = S.Material(color=FLESH, glow=0.35)
    span = (arc[1] - arc[0]) / count
    for k in range(count):
        a = math.radians(arc[0] + span * (k + 0.5 + offset))
        r = radius * 0.86
        w = span * math.pi / 180 * r * 0.42
        big = 1.0 if k % 2 == 0 else 0.7
        L = length * big * (1 + crooked * rng.uniform(-0.3, 0.3))
        c = np.array([math.sin(a) * r, math.cos(a) * r, 0.0])
        t = np.array([math.cos(a), -math.sin(a), 0.0])  # along the rim
        inward = -np.array([math.sin(a), math.cos(a), 0.0]) * 0.9
        tip = c + inward * 0.4 + np.array([0, 0, z_sign * L]) + t * crooked * rng.uniform(-0.4, 0.4)
        pts = [c - t * w, c + t * w, c + inward * 1.0, tip]
        out.append(S.hull(pts, tooth))
    return out


def _stem(top: float, height: float, lean: float, thickness: float, leaf_team: bool, seed: int = 0) -> list:
    green = P.mottled(P.STEM, P.STEM * 0.6, scale=1.0, amount=0.3, seed=seed)
    pts = [np.array([0, 0, top - 0.6]),
           np.array([lean * 0.2, -lean * 0.1, top + height * 0.45]),
           np.array([lean * 0.7, -lean * 0.25, top + height * 0.85]),
           np.array([lean * 1.25, -lean * 0.2, top + height * 1.0])]
    out = []
    for k in range(3):
        r0 = thickness * (1.0 - 0.18 * k)
        out += S.limb(pts[k], pts[k + 1], r0, green, r0 * 0.82)
    if leaf_team:  # a leaf in the owner's colour: who he belongs to
        leaf = S.Ellipsoid(P.team(0.62), S._orient(pts[1] + np.array([-thickness * 1.6, 0.3, -0.4]), (0, 25, -30)),
                           radii=(thickness * 2.2, thickness * 1.1, 0.35))
        out.append(leaf)
    return out


@dataclasses.dataclass
class PumpkinLook:
    base: np.ndarray = dataclasses.field(default_factory=lambda: P.PUMPKIN)
    dark: np.ndarray = dataclasses.field(default_factory=lambda: P.PUMPKIN_DARK)
    rot: float = 0.15
    glow: np.ndarray = dataclasses.field(default_factory=lambda: P.CANDLE)
    eyes: tuple = (((11, 34), (46, 52), (30, 12)),)  # the right eye; mirrored for the left
    nose: tuple = ()
    teeth: int = 9
    tooth_length: float = 2.4
    crooked: float = 0.0
    squash: float = 0.9  # height / width
    seed: int = 0


SCARY = PumpkinLook()
# Pac-Man himself: a carved jack-o'-lantern rather than a ball cut in half (the player asked, 2026-10-24): a little
# taller than wide, deep ribs, angry eyes, a carved nose, crooked teeth, the cut low under the nose
JACK = PumpkinLook(rot=0.25, eyes=(((10, 32), (46, 50), (32, 10)),), nose=(((-7, 2), (7, 2), (0, 15)),), teeth=7,
                   tooth_length=2.2, crooked=0.9, squash=1.02, seed=3)
SCARIER = PumpkinLook(base=P.hexrgb("#d9640e"), dark=P.hexrgb("#8a3206"), rot=0.9, glow=P.hexrgb("#ffb030"),
                      eyes=(((12, 30), (50, 50), (34, 6)),), nose=(((-6, 2), (6, 2), (0, 15)),), teeth=11,
                      tooth_length=3.0, crooked=1.0, squash=0.86, seed=7)


def brain(seed: int = 0) -> S.Material:
    """A brain: pink-grey and glistening, folded in winding ridges with dark grooves between them."""
    def color(look):
        p = look.p
        g = np.abs(np.sin(p[:, 0] * 2.6 + p[:, 1] * 1.4 + P.noise(p, 1.1, seed) * 5.0))
        c = P.mix(P.hexrgb("#e2a0aa"), P.hexrgb("#a45a6a"), P.fbm(p, 1.6, seed + 1) * 0.6)
        return np.where((g < 0.28)[:, None], P.hexrgb("#4a1420"), c)
    return S.Material(color=color, glow=0.18, shine=0.7, gloss=30, vivid=0.85)


def pumpkin_head(look: PumpkinLook = SCARY, radius: float = R, centre: float = R + 1.0, team_leaf: bool = True,
                 vines: bool = False, split: float = 0.0, with_brain: bool = False) -> tuple[Part, Part, Part]:
    """The two jaws and the burst, around a pumpkin centred `centre` above the ground. `split` lowers the mouth's
    cut below the middle (a fraction of the radius); `with_brain` puts a brain in the mouth instead of the candle."""
    feats = [tri for e in look.eyes for tri in (e, tuple((-u, v) for u, v in e))] + list(look.nose)
    skin = face_material(feats, look.base, look.dark, look.rot, look.seed, look.glow)
    plain = face_material([], look.base, look.dark, look.rot, look.seed + 1, look.glow)
    radii = (radius * 1.06, radius, radius * look.squash)
    s = split * radius  # the cut, this far below the middle
    rim = radius * math.sqrt(max(0.05, 1 - (s / radii[2]) ** 2))  # the pumpkin's radius at the cut
    hollow = inside(rim, look.glow)
    skin.cut = hollow
    plain.cut = hollow
    zc = centre - s
    upper = S.Ellipsoid(skin, S._orient((0, 0, centre)), radii=radii).clip((0, 0, -1), s)
    lower = S.Ellipsoid(plain, S._orient((0, 0, centre)), radii=radii).clip((0, 0, 1), -s)
    top_teeth = [_moved(t, (0, 0, zc)) for t in _teeth(rim, -1, look.teeth, look.tooth_length,
                                                       seed=look.seed, crooked=look.crooked)]
    low_teeth = [_moved(t, (0, 0, zc)) for t in _teeth(rim, 1, look.teeth - 1, look.tooth_length * 0.85,
                                                       arc=(-68, 68), offset=0.0, seed=look.seed + 1,
                                                       crooked=look.crooked)]
    if with_brain:  # a brain lying in the hollow of the lower jaw, lit from below
        bm = brain(look.seed)
        inner = [S.Ellipsoid(bm, S._orient((sx * rim * 0.24, rim * 0.08, zc + rim * 0.2), (0, sx * 8, 0)),
                             radii=(rim * 0.27, rim * 0.5, rim * 0.3)) for sx in (1, -1)]
        inner.append(S.Ellipsoid(bm, S._orient((0, -rim * 0.12, zc + rim * 0.06)), radii=(rim * 0.32, rim * 0.34,
                                                                                         rim * 0.2)))
    else:
        flame = S.Ellipsoid(S.mat(P.hexrgb("#fff2a8"), glow=1.0), S._orient((0, 0, zc + 1.2)), radii=(0.9, 0.9, 1.8))
        flame.casts = False
        inner = [S.rod((0, 0, zc - 0.2), (0, 0, zc + 0.6), 0.9, S.mat(P.hexrgb("#efe6c8"), glow=0.3)), flame]
    upper_part = Part("jaw_top", pivot=(0, 0, zc), boxes=[upper, *top_teeth,
                                                           *_stem(centre + radii[2] - 0.2, radius * 0.45,
                                                                  radius * 0.18, radius * 0.13, team_leaf,
                                                                  look.seed)])
    lower_part = Part("jaw_bottom", pivot=(0, 0, zc), boxes=[lower, *low_teeth, *inner])
    if vines:
        lower_part.boxes += _vines(radius, centre, look.seed)
    shards = []
    rng = np.random.default_rng(look.seed + 99)
    for k in range(14):
        d = rng.normal(0, 1, 3)
        d[2] = abs(d[2]) * 0.8 + 0.2
        d /= np.linalg.norm(d)
        c = np.array([0, 0, centre]) + d * radius * rng.uniform(0.9, 1.5)
        size = radius * rng.uniform(0.12, 0.22)
        pts = c + rng.normal(0, size, (5, 3))
        shards.append(S.hull(pts, S.mat(P.CANDLE if k % 3 else P.PUMPKIN, glow=0.9 if k % 3 else 0.2)))
    pop = Part("pop", boxes=shards)
    return upper_part, lower_part, pop


def _moved(shape, by):
    shape.xf = S.affine(None, by) @ shape.xf
    return shape


def _vines(radius: float, centre: float, seed: int) -> list:
    """Dark green vines creeping up from the ground and round the lower half, with a few leaves."""
    rng = np.random.default_rng(seed + 5)
    green = P.mottled(P.hexrgb("#3d5a22"), P.hexrgb("#22331a"), scale=1.5, amount=0.4, seed=seed)
    leaf = P.mottled(P.hexrgb("#4f7a2a"), P.hexrgb("#2a4418"), scale=1.0, amount=0.3, seed=seed + 1)
    out = []
    for v in range(4):
        a0 = rng.uniform(0, 2 * math.pi)
        pts = []
        for k in range(7):
            a = a0 + k * 0.35 * (1 if v % 2 else -1)
            el = -1.2 + k * 0.17
            r = radius * 1.04
            pts.append(np.array([math.cos(a) * math.cos(el) * r * 1.06, math.sin(a) * math.cos(el) * r,
                                 centre + math.sin(el) * r * 0.9]))
        for a, b in zip(pts, pts[1:]):
            out += S.limb(a, b, 0.55, green)
        for k in (2, 4):
            out.append(S.Ellipsoid(leaf, S._orient(pts[k] + rng.normal(0, 0.5, 3), (rng.uniform(0, 90), 0,
                                                                                     rng.uniform(0, 180))),
                                   radii=(1.6, 0.9, 0.3)))
    return out


def pacman(key: str = "pacman") -> Unit:
    """Pac-Man: a carved jack-o'-lantern, its mouth cut low under the nose, a brain in it instead of a candle."""
    upper, lower, pop = pumpkin_head(JACK, split=0.38, with_brain=True)
    root = Part("root").add(Part("body").add(upper, lower), pop)
    return Unit(key, "Pac-Man", "Furious the Monkey Boy (cheat)", "pacman", root, group="easter",
                attack="chomp", scale=2.4)


GIANT_SCALE = 30.0  # as in Age of Minecraft: wider and taller than a Castle and a Wonder side by side


def giant_pacman(key: str = "giant_pacman") -> Unit:
    """The giant Pac-Man: an older, rotting pumpkin, vines creeping over it, a meaner grin and more teeth."""
    upper, lower, pop = pumpkin_head(SCARIER, team_leaf=False, vines=True)
    root = Part("root").add(Part("body").add(upper, lower), pop)
    return Unit(key, "Giant Pac-Man", "Saboteur (cheat)", "pacman", root, group="easter", attack="chomp",
                scale=GIANT_SCALE)


def lantern(at, radius: float = 3.0, look: PumpkinLook = SCARY, seed: int = 0) -> list:
    """A small jack-o'-lantern sitting on the ground (or a ledge) at `at`, facing +y: for porches and fields."""
    feats = [tri for e in look.eyes for tri in (e, tuple((-u, v) for u, v in e))] + [((-40, -8), (40, -8), (0, -30))]
    skin = face_material(feats, look.base, look.dark, look.rot, seed, look.glow)
    c = np.asarray(at, float) + (0, 0, radius * 0.85)
    body = S.Ellipsoid(skin, S._orient(c), radii=(radius * 1.08, radius, radius * 0.85))
    stem = S.limb(c + (0, 0, radius * 0.75), c + (radius * 0.12, 0, radius * 1.15), radius * 0.15,
                  P.mottled(P.STEM), radius * 0.11)
    return [body, *stem]
