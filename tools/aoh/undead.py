"""The undead: skeleton soldiers, built on the same joints as Age of Minecraft's people, so the rigs in animation.py
move them unchanged (legs from the hips, arms from the shoulders, the head on the neck, what they hold in the
hands as "item_r" / "item_l").

Proportions are a cartoon's: a big skull, thin bones with knobbly joints, a ribcage, and torn cloth in the owner's
colour so you can tell whose they are. Model units as everywhere: a soldier stands about 30 high and faces +y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import paint as P
from . import sculpt as S
from .geometry import Part
from .units import Unit

HIP = 12.0  # hip joint height (the leg pivots), as the biped rig expects
SHOULDER = 22.0  # shoulder pivot height
NECK = 24.0
ARM_X = 4.6  # shoulder joints' distance from the middle
LEG_X = 2.0


def _skull_material(glow=P.hexrgb("#9cff6a"), seed: int = 0) -> S.Material:
    """Bone with deep eye sockets, each holding a small glowing light, a nose hole and a row of teeth."""
    b = P.bone(seed)

    def socket(p):
        out = np.zeros(len(p), bool)
        pupil = np.zeros(len(p), bool)
        for sx in (-1, 1):
            dx = (p[:, 0] - sx * 1.3) / 1.15
            dz = (p[:, 2] + 0.1) / 1.05
            r = dx * dx + dz * dz
            out |= (r < 1) & (p[:, 1] > 0.5)
            pupil |= (r < 0.16) & (p[:, 1] > 0.5)
        nose = (np.abs(p[:, 0]) < 0.45 - (p[:, 2] + 1.6) * 0.25) & (p[:, 2] < -1.0) & (p[:, 2] > -1.9) & (p[:, 1] > 0)
        return out, pupil, nose

    def color(look):
        c = b.color(look)
        s, pupil, nose = socket(look.p)
        c = np.where((s | nose)[:, None], P.hexrgb("#1a1018"), c)
        return np.where(pupil[:, None], np.asarray(glow, float), c)

    def glow_f(look):
        return socket(look.p)[1].astype(float)

    return S.Material(color=color, glow=glow_f, shine=0.2, gloss=12, vivid=0.5)


def _jaw_material(seed: int = 0) -> S.Material:
    b = P.bone(seed)

    def color(look):
        c = b.color(look)
        p = look.p
        teeth = (p[:, 2] > -0.2) & (p[:, 1] > 0.3)
        gaps = teeth & (np.abs(np.sin(p[:, 0] * 3.2)) < 0.25)
        line = (np.abs(p[:, 2] + 0.15) < 0.12) & (p[:, 1] > 0.3)
        return np.where((gaps | line)[:, None], P.hexrgb("#2a1c1c"), c)
    return S.Material(color=color, shine=0.2, gloss=12, vivid=0.5)


def _ribs_material(seed: int = 0) -> S.Material:
    """A ribcage: bands of bone with dark gaps, a breastbone down the middle."""
    b = P.bone(seed)

    def color(look):
        p = look.p
        c = b.color(look)
        band = np.abs(np.sin(p[:, 2] * 2.4 + np.abs(p[:, 0]) * 0.35)) < 0.42
        sternum = (np.abs(p[:, 0]) < 0.6) & (p[:, 1] > 0)
        dark = band & ~sternum
        return np.where(dark[:, None], P.hexrgb("#2b1f26"), c)
    return S.Material(color=color, shine=0.15, gloss=10, vivid=0.5)


@dataclass
class Gear:
    helmet: Optional[str] = None  # None, "kettle", "nasal", "great", "hood"
    cloth: str = "loin"  # "loin", "tabard", "robe"
    shield: Optional[str] = None  # None, "round", "kite"
    right: Optional[str] = "sword"  # what the right hand holds
    left: Optional[str] = None
    armour: bool = False  # a rusty breastplate over the ribs
    eyes: tuple = (0.61, 1.0, 0.42)  # the glow in the eye sockets


def _bone_limb(a, b, r0, r1, m) -> list:
    """A bone: a thin shaft with knobbly ends."""
    r0, r1 = r0 * 1.3, r1 * 1.3  # chunky enough to read at the game's size
    return [S.rod(a, b, r0, m, r1), S.ball(a, r0 * 1.45, m), S.ball(b, r1 * 1.45, m)]


def _sword(m_blade, m_hilt, length=11.0) -> list:
    """A rusty sword in the hand at the origin, pointing forward and a little up (as the rig holds items)."""
    d = np.array([0.0, math.cos(math.radians(20)), math.sin(math.radians(20))])
    up = np.array([0.0, -math.sin(math.radians(20)), math.cos(math.radians(20))])
    base = d * 1.6
    tip = base + d * length
    w, t = 0.85, 0.22
    blade = S.hull([base + up * w + [t, 0, 0], base - up * w + [t, 0, 0], base + up * w - [t, 0, 0],
                    base - up * w - [t, 0, 0], tip - d * 1.8 + up * w * 0.8, tip - d * 1.8 - up * w * 0.8, tip],
                   m_blade)
    guard = S.block((-0.5, -0.35, -2.4), (0.5, 0.35, 2.4), m_hilt, rot=(20, 0, 0), at=tuple(base))
    grip = S.rod(-d * 1.4, base, 0.45, P.wood(P.hexrgb("#4a2c1c"), P.hexrgb("#2a1810")))
    pommel = S.ball(-d * 1.6, 0.7, m_hilt)
    return [blade, guard, grip, pommel]


def _round_shield(radius=4.2) -> list:
    """A round wooden shield, painted in the owner's colour with a dark cross, an iron boss and rim, facing out to
    the left (-x) from the left hand."""
    def pattern(look):
        p = look.p
        cross = (np.abs(p[:, 0]) < 0.6) | (np.abs(p[:, 1]) < 0.6)
        return np.where(cross, 0.45, 1.0)
    face = P.team(0.66, pattern)
    board = S.rod((0.3, 0, 0), (-0.6, 0, 0), radius, face)
    rim = S.torus((-0.6, 0, 0), radius, 0.35, P.metal(rust=0.6), rot=(0, 90, 0))
    boss = S.ball((-0.7, 0, 0), (0.6, 1.2, 1.2), P.metal(rust=0.3))
    return [board, rim, boss]


def _helmet(kind: str) -> list:
    iron = P.metal(P.hexrgb("#9a9aa6"), rust=0.5, seed=3)
    top = NECK + 3.6
    if kind == "kettle":  # a wide-brimmed iron hat
        return [S.ball((0, 0.3, top + 0.4), (3.7, 3.9, 2.8), iron).clip((0, 0, -1), -0.2),
                S.rod((0, 0.3, top + 0.1), (0, 0.3, top + 0.5), 5.4, iron)]
    if kind == "nasal":
        return [S.ball((0, 0.3, top + 0.2), (3.75, 3.95, 3.6), iron).clip((0, 0, -1), 0.3),
                S.block((-0.35, 3.8, top - 2.6), (0.35, 4.4, top + 0.6), iron)]
    if kind == "great":  # a pot helm with a slit
        return [S.rod((0, 0.3, top - 4.2), (0, 0.3, top + 1.6), 3.9, iron, 3.6),
                S.block((-2.6, 3.7, top - 1.6), (2.6, 4.4, top - 1.0), S.mat(P.hexrgb("#120c10")))]
    if kind == "hood":
        return [S.ball((0, -0.3, top - 0.6), (4.2, 4.4, 4.4), P.team(0.38)).clip((0, 1, 0), 1.2)]
    return []


def skeleton(key: str, name: str, replaces: str, gear: Gear = Gear(), attack: str = "chop", group: str = "foot",
             seed: int = 0, scale: float = 1.0) -> Unit:
    b = P.bone(seed)
    rust = P.metal(P.hexrgb("#9696a2"), rust=0.6, seed=seed)
    hilt = P.metal(P.hexrgb("#5a4a3a"), rust=0.4, seed=seed + 1)

    # legs: thigh and shin, a knee, a foot
    def leg(name, x):
        knee = (x, 0.3, 6.5)
        ankle = (x, 0.0, 1.4)
        shapes = _bone_limb((x, 0, HIP - 0.4), knee, 0.62, 0.5, b) + _bone_limb(knee, ankle, 0.5, 0.42, b)
        shapes.append(S.ball((x, 1.0, 0.7), (1.0, 1.9, 0.75), b))
        return Part(name, pivot=(x, 0, HIP), boxes=shapes)

    pelvis = S.ball((0, 0, HIP + 0.4), (3.2, 2.0, 1.5), b)
    spine = [S.ball((0, -0.8, z), (0.75, 0.75, 0.55), b) for z in np.linspace(HIP + 1.6, SHOULDER - 2.0, 5)]
    ribs = S.ball((0, 0.2, SHOULDER - 3.2), (3.6, 2.7, 3.9), _ribs_material(seed)).clip((0, 0, -1), 2.6)
    collar = S.rod((-ARM_X + 0.6, 0.1, SHOULDER + 0.6), (ARM_X - 0.6, 0.1, SHOULDER + 0.6), 0.55, b)
    torso_shapes = [pelvis, *spine, ribs, collar]
    if gear.armour:
        torso_shapes.append(S.ball((0, 0.4, SHOULDER - 3.4), (3.95, 3.0, 3.9), rust).clip((0, 0, -1), 3.0)
                            .clip((0, -1, 0), 0.5))
    if gear.cloth == "loin":  # a torn cloth hanging from the hips, in the owner's colour
        torso_shapes.append(S.rod((0, 0, HIP + 1.2), (0, 0.15, HIP - 4.2), 3.5, P.team(0.6, _torn), 4.1,
                                  capped=False))
    elif gear.cloth == "tabard":
        torso_shapes.append(S.block((-3.2, 2.1, HIP - 4.5), (3.2, 2.6, SHOULDER - 0.5), P.team(0.62, _torn)))
        torso_shapes.append(S.block((-3.2, -3.0, HIP - 4.5), (3.2, -2.5, SHOULDER - 0.5), P.team(0.5, _torn)))
    elif gear.cloth == "robe":
        torso_shapes.append(S.rod((0, 0, SHOULDER + 0.5), (0, 0, 0.6), 3.3, P.team(0.55, _torn), 5.2))

    def arm(name, x, sign):
        elbow = (x + sign * 0.3, 0.4, SHOULDER - 5.0)
        hand = (x + sign * 0.2, 0.9, SHOULDER - 9.4)
        shapes = [S.ball((x, 0, SHOULDER), 1.0, b)] + _bone_limb((x, 0, SHOULDER - 0.3), elbow, 0.55, 0.45, b) \
            + _bone_limb(elbow, hand, 0.45, 0.38, b) + [S.ball(hand, (0.75, 0.9, 1.0), b)]
        return Part(name, pivot=(x, 0, SHOULDER), boxes=shapes), np.array(hand)

    arm_r, hand_r = arm("arm_r", ARM_X, 1)
    arm_l, hand_l = arm("arm_l", -ARM_X, -1)
    if gear.right == "sword":
        arm_r.add(Part("item_r", pivot=tuple(hand_r), boxes=[_moved(s, hand_r) for s in _sword(rust, hilt)]))
    if gear.shield == "round":
        arm_l.add(Part("item_l", pivot=tuple(hand_l), boxes=[_moved(s, hand_l + (-0.9, 0.6, 0.6))
                                                             for s in _round_shield()]))

    head = Part("head", pivot=(0, 0, NECK), boxes=[
        S.rod((0, -0.6, NECK - 1.6), (0, -0.3, NECK + 0.6), 0.6, b),
        S.Ellipsoid(_skull_material(gear.eyes, seed), S._orient((0, 0.4, NECK + 3.6)), radii=(3.4, 3.6, 3.3)),
        S.Ellipsoid(_jaw_material(seed), S._orient((0, 1.7, NECK + 1.0)), radii=(2.3, 2.1, 1.25)),
    ] + (_helmet(gear.helmet) if gear.helmet else []))
    torso = Part("torso", pivot=(0, 0, HIP), boxes=torso_shapes).add(head, arm_r, arm_l)
    root = Part("root").add(torso, leg("leg_r", LEG_X), leg("leg_l", -LEG_X))
    return Unit(key, name, replaces, "biped", root, group=group, attack=attack, scale=scale)


def _torn(look):
    """Torn hems: ragged holes near the bottom of hanging cloth, and darker folds."""
    p = look.p
    folds = 0.82 + 0.18 * np.sin(np.arctan2(p[:, 1], p[:, 0]) * 7)
    return folds * (0.8 + 0.2 * P.noise(p, 1.5, 11))


def _moved(shape, by):
    shape.xf = S.affine(None, tuple(np.asarray(by, float))) @ shape.xf
    return shape


def militia(key: str = "militia") -> Unit:
    return skeleton(key, "Skeleton Militia", "Militia", Gear(helmet=None, cloth="loin", right="sword",
                                                             shield="round"))


def man_at_arms(key: str = "man_at_arms") -> Unit:
    return skeleton(key, "Skeleton Man-at-Arms", "Man-at-Arms",
                    Gear(helmet="nasal", cloth="tabard", right="sword", shield="round", armour=True), seed=1)
