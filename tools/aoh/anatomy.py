"""Bodies that look alive, not robotic (an 8-year-old tester found the units "robotic", 2026-10-24): limbs of
muscle tapering to slim wrists and ankles with no ball joints showing, hands with fingers and a thumb (claws on the
monsters), feet and boots with heels and toes, a torso of chest, waist and hips with shoulders and a neck, and
clothes that hang: sleeves, trousers, skirts that flare and end in torn strips, belts with buckles and pouches.
Skeletons get thin bones with knobbly ends, shoulder blades, a pelvis and rags.

Everything here builds shapes in creatures.py's body frame (facing +y, feet at z 0); creatures.person puts them on
the rig's parts.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import paint as P
from . import sculpt as S

FINE = dict(casts=False, ink=False)  # small details: no shadow, no outline (an outline round each finger is a seam)


def _fine(shape):
    shape.casts = False
    shape.ink = False
    return shape


def soft(shapes: list, label: str) -> list:
    """The shapes melt together where they meet (sculpt.BLEND): one body, not a doll's jointed pieces."""
    for s in shapes:
        if s.blend is None:
            s.blend = label
    return shapes


def _v(p) -> np.ndarray:
    return np.asarray(p, float)


# --------------------------------------------------------------------------- limbs

def muscle(a, c, r0: float, r1: float, m: S.Material, belly: float = 0.38, bulge: float = 1.18,
           back: float = 0.0) -> list:
    """A limb segment of flesh from a to c: a core tapering from r0 to r1, and a muscle swelling along it (its
    widest `belly` of the way down, `bulge` times the thicker end, moved `back` toward -y for a calf)."""
    a, c = _v(a), _v(c)
    L = float(np.linalg.norm(c - a))
    if L < 1e-6:
        return []
    rb = max(r0, r1) * bulge
    belly_shape = S.Ellipsoid(m, S.axis_frame(a, c) @ S.affine(None, (0.0, -back, L * belly)),
                              radii=(rb, rb * 0.92, L * 0.44))
    return soft([S.rod(tuple(a), tuple(c), r0, m, r1), S.ball(tuple(c), r1 * 1.02, m), belly_shape], "flesh")


def bone(a, c, r: float, m: S.Material, knob: float = 1.5) -> list:
    """A bone: a thin shaft, a little thicker at its ends, each end two small knobs side by side (the joint)."""
    a, c = _v(a), _v(c)
    d = c - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return []
    side = np.cross(d / L, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 * L else [1.0, 0.0, 0.0])
    side = side / max(1e-6, np.linalg.norm(side)) * r * 0.55
    out = [S.rod(tuple(a), tuple(a + d * 0.5), r * 1.15, m, r * 0.8), S.rod(tuple(a + d * 0.5), tuple(c), r * 0.8, m,
                                                                          r * 1.15)]
    for end in (a, c):
        out += [S.ball(tuple(end + side), r * knob * 0.6, m), S.ball(tuple(end - side), r * knob * 0.6, m)]
    return soft(out, "bone")


# --------------------------------------------------------------------------- hands and feet

def hand(at, sign: int, m: S.Material, claws: bool = False, bony: bool = False, glove: Optional[S.Material] = None
         ) -> list:
    """A hand hanging at `at` (its fingers curling forward, +y): a palm, four fingers and a thumb; bones for a
    skeleton, long pale claws for the monsters, a gauntlet's cuff with `glove`."""
    at = _v(at)
    out = []
    pm = glove or m
    r = 0.13 if bony else 0.17
    if not bony:
        out.append(S.Ellipsoid(pm, S._orient(tuple(at + (0, 0.1, -0.15))), radii=(0.55, 0.42, 0.68)))
    else:
        out.append(S.Ellipsoid(m, S._orient(tuple(at + (0, 0.1, -0.15))), radii=(0.42, 0.28, 0.5)))
    claw_m = S.mat(P.hexrgb("#e2d8c0"), shine=0.3)
    for k in range(4):
        x = (k - 1.5) * 0.24
        base = at + (x, 0.3, -0.62)
        mid = base + (x * 0.15, 0.32, -0.42)
        tip = mid + (0, 0.25, -0.3 if not claws else -0.55)
        out += [_fine(S.rod(tuple(base), tuple(mid), r, pm, r * 0.9)),
                _fine(S.rod(tuple(mid), tuple(tip), r * 0.9, claw_m if claws else pm, 0.04 if claws else r * 0.75))]
    thumb0 = at + (-sign * 0.42, 0.25, -0.15)
    out.append(_fine(S.rod(tuple(thumb0), tuple(thumb0 + (-sign * 0.2, 0.42, -0.42)), r * 1.05, pm, r * 0.8)))
    if glove is not None:
        out.append(S.rod(tuple(at + (0, 0, 0.55)), tuple(at + (0, 0.05, 0.05)), 0.62, glove, 0.55))
    return soft(out, "bone" if bony else "flesh")


def foot(x: float, m: S.Material, kind: str = "bare", bony: bool = False, claws: bool = False) -> list:
    """A foot at (x, 0) under the ankle: a heel and a forefoot reaching forward (+y); toes when bare, a boot's
    shaft and sole when shod, long toe bones on a skeleton."""
    out = []
    if bony:
        out += [S.Ellipsoid(m, S._orient((x, -0.2, 0.55)), radii=(0.45, 0.55, 0.45))]
        for k in range(4):
            dx = (k - 1.5) * 0.28
            out.append(_fine(S.rod((x + dx * 0.6, 0.2, 0.5), (x + dx, 1.5, 0.2), 0.14, m, 0.11)))
        return soft(out, "bone")
    out += [S.Ellipsoid(m, S._orient((x, -0.35, 0.62)), radii=(0.62, 0.68, 0.6)),
            S.Ellipsoid(m, S._orient((x, 0.75, 0.42)), radii=(0.68, 1.15, 0.42))]
    if kind == "boot":
        sole = S.mat(P.hexrgb("#1a1210"))
        out += [S.rod((x, 0.0, 0.8), (x, 0.1, 4.2), 0.78, m, 0.72), _fine(S.Ellipsoid(sole, S._orient((x, 0.4, 0.08)),
                                                                                      radii=(0.72, 1.55, 0.14)))]
    elif kind == "bare":
        toe_m = S.mat(P.hexrgb("#e2d8c0"), shine=0.3) if claws else m
        for k in range(3):
            dx = (k - 1) * 0.36
            out.append(_fine(S.Ellipsoid(toe_m if claws else m, S._orient((x + dx, 1.75 + (0.3 if claws else 0), 0.3)),
                                         radii=(0.17, 0.32 if not claws else 0.5, 0.15))))
    return soft(out, "flesh")


# --------------------------------------------------------------------------- the torso

TORSO = {  # chest width, depth; waist; how full the belly is
    "werewolf": (4.0, 2.8, 0.82, 1.0), "imp": (2.8, 2.0, 0.78, 1.0), "ghoul": (2.9, 1.9, 0.68, 0.9),
    "vampire": (3.2, 2.0, 0.72, 0.9), "zombie": (3.2, 2.2, 0.85, 1.15), "draugr": (3.5, 2.4, 0.86, 1.0),
    "mummy": (3.1, 2.0, 0.78, 0.9), "jiangshi": (3.1, 2.1, 0.8, 1.0),
}


def torso(kind: str, skin: S.Material, hip: float, sh: float, arm_x: float) -> list:
    """A flesh torso: a chest narrower front to back than across, a waist, the hips, rounded shoulders where the
    arms join, the slope of the neck muscles, a neck."""
    cw, cd, waist, belly = TORSO.get(kind, (3.3, 2.2, 0.8, 1.0))
    out = [S.Ellipsoid(skin, S._orient((0, 0.15, sh - 3.0)), radii=(cw, cd, 3.3)),
           S.Ellipsoid(skin, S._orient((0, 0.15 + 0.3 * (belly - 1), hip + 3.4)), radii=(cw * waist, cd * 0.9 * belly,
                                                                                      2.6)),
           S.Ellipsoid(skin, S._orient((0, -0.05, hip + 1.0)), radii=(cw * 0.88, cd * 0.95, 2.0)),
           S.Ellipsoid(skin, S._orient((0, -0.35, sh + 0.1)), radii=(cw * 0.75, cd * 0.7, 1.0))]
    for sx in (1, -1):  # the slope from the neck out to the shoulder (the shoulder's round is the arm's deltoid)
        out.append(S.Ellipsoid(skin, S._orient((sx * (arm_x - 1.5), -0.1, sh - 0.4), (0, sx * 18, 0)),
                               radii=(1.9, 1.35, 1.0)))
    return soft(out, "flesh")


def deltoid(shoulder, sign: int, r: float, m: S.Material) -> list:
    """The shoulder's muscle, on the arm (so it turns with it): a long rounded cap over the top of the upper arm,
    not a ball joint."""
    at = _v(shoulder) + (-sign * 0.25, -0.05, -0.9 * r)
    return soft([S.Ellipsoid(m, S._orient(tuple(at), (0, -sign * 12, 0)), radii=(r * 1.12, r * 1.15, r * 1.75))],
                "flesh")


def skeleton_torso(m: S.Material, ribs_m: S.Material, hip: float, sh: float, arm_x: float) -> list:
    """A skeleton's trunk: a pelvis, a spine of vertebrae, the ribcage, collarbones and shoulder blades."""
    out = [S.Ellipsoid(m, S._orient((0, -0.1, hip + 0.4)), radii=(2.9, 1.6, 1.4)).clip((0, 1, 0), 1.0),
           S.Ellipsoid(m, S._orient((0, -0.1, hip + 0.4)), radii=(2.9, 1.6, 1.4)).clip((0, -1, 0), 0.2)]
    for k, z in enumerate(np.linspace(hip + 1.6, sh - 2.0, 6)):
        out.append(S.Ellipsoid(m, S._orient((0, -0.9, z)), radii=(0.62, 0.6, 0.42)))
    out.append(S.ball((0, 0.2, sh - 3.2), (3.3, 2.4, 3.6), ribs_m).clip((0, 0, -1), 2.6))
    for sx in (1, -1):
        out += [S.rod((sx * 0.6, 0.6, sh + 0.4), (sx * (arm_x - 0.4), 0.2, sh + 0.2), 0.32, m, 0.28),  # collarbone
                S.Ellipsoid(m, S._orient((sx * 2.0, -1.9, sh - 1.8), (0, 0, sx * 10)), radii=(1.4, 0.35, 1.9))]
    return soft(out, "bone")


# --------------------------------------------------------------------------- clothes

def drape(m: S.Material, top: float, bottom: float, r_top: float, r_bottom: float, seed: int = 0,
          tatters: int = 9, front_slit: bool = False, y_scale: float = 0.78) -> list:
    """A skirt or a robe's lower half: cloth flaring from `top` to `bottom`, oval (deeper `y_scale`), its hem torn
    into strips of different lengths hanging below it."""
    rng = np.random.default_rng(seed + 900)
    n = 16
    ring = lambda r, z: [(math.cos(2 * math.pi * k / n) * r, math.sin(2 * math.pi * k / n) * r * y_scale, z)
                         for k in range(n)]
    mid_r = r_top + (r_bottom - r_top) * 0.45
    upper = S.hull(ring(r_top, top) + ring(mid_r, top - (top - bottom) * 0.45), m)
    body = S.hull(ring(mid_r, top - (top - bottom) * 0.45) + ring(r_bottom, bottom), m)
    out = [upper, body]
    for k in range(tatters):
        a = 2 * math.pi * (k + rng.uniform(-0.25, 0.25)) / tatters
        if front_slit and abs(math.sin(a) - 1.0) < 0.15:
            continue
        L = rng.uniform(0.6, 2.4)
        w = 2 * math.pi * r_bottom / tatters * 0.36
        c = np.array([math.cos(a) * r_bottom * 0.98, math.sin(a) * r_bottom * y_scale * 0.98, bottom + 0.05])
        t = np.array([-math.sin(a), math.cos(a), 0.0])
        out.append(_fine(S.hull([c - t * w, c + t * w, c - t * w * 0.6 + (0, 0, -L * 0.6) + c * 0.04,
                                 c + t * w * 0.5 + (0, 0, -L * 0.7) + c * 0.04, c + (0, 0, -L) + c * 0.06,
                                 c - t * w + _v((0.0, 0.0, 0.25)), c + t * w + _v((0.0, 0.0, 0.25))], m)))
    return soft(out, "cloth")


def belt(z: float, r: float, m: S.Material, seed: int = 0, pouch: bool = True, y_scale: float = 0.72) -> list:
    """A belt round the waist, a metal buckle at the front, a pouch on the hip."""
    buckle = P.metal(P.hexrgb("#8a6a3a"), rust=0.4, seed=seed)
    out = [S.Ellipsoid(m, S._orient((0.0, 0.0, z)), radii=(r + 0.15, r * y_scale + 0.15, 1.4))
           .clip((0, 0, 1), 0.45).clip((0, 0, -1), 0.45),
           _fine(S.block((-0.55, r * y_scale - 0.05, z - 0.42), (0.55, r * y_scale + 0.3, z + 0.42), buckle))]
    if pouch:
        leather = S.mat(P.hexrgb("#4a3020"), shine=0.1)
        out.append(S.Ellipsoid(leather, S._orient((r * 0.82, 0.6, z - 1.0)), radii=(0.6, 0.75, 0.9)))
    return out


def sleeve(a, c, r: float, m: S.Material, flare: float = 1.25) -> list:
    """A sleeve over the upper arm (from the shoulder a toward the elbow c), loose and a little wider at its cuff."""
    a, c = _v(a), _v(c)
    end = a + (c - a) * 0.92
    return soft([S.rod(tuple(a + (c - a) * 0.08), tuple(end), r, m, r * flare),
                 S.Ellipsoid(m, S.axis_frame(a, c) @ S.affine(None, (0.0, 0.0, r * 0.6)), radii=(r, r, r * 1.3))],
                "cloth")


def trouser(a, c, r0: float, r1: float, m: S.Material) -> list:
    """Loose trousers over a leg segment, ragged at the shin."""
    a, c = _v(a), _v(c)
    return soft([S.rod(tuple(a), tuple(c), r0, m, r1)], "cloth")
