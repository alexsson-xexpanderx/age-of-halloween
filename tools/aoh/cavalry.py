"""Mounted units: undead riders on undead steeds.

Steeds use the horse rig's parts (animation.py "cavalry" / "horse_archer"): "horse_body" (with the rider, "horse_head"
and "tail" on it) and "leg_fr", "leg_fl", "leg_br", "leg_bl" pivoting at the hips and shoulders. Riders are
creatures.py people seated as Age of Minecraft seated its riders (mounted.mount).

    nightmare   a black horse, flaming mane and tail, burning eyes and hooves

Every horse is modelled like a horse (the player found the old ones looked like balloons): a barrel, a deep chest, a
broad croup, withers, shoulder and thigh muscles, an arched neck with a mane falling to one side, a long head (a
broad forehead and round jowl tapering to a muzzle with nostrils, eyes on the sides, pointed ears, a forelock), legs
with knees or hocks, slim cannons, fetlocks, angled pasterns and hooves, a tail of flowing locks; a saddle with a
pommel, a cantle and stirrups on a blanket in the owner's colour, a bridle and reins.
    bonehorse   a horse's skeleton
    ghosthorse  a pale spectral horse, glowing
    mummycamel  a camel wrapped in bandages
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np

from . import anatomy as A
from . import creatures as C
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .creatures import Gear, body, person
from .geometry import Part
from .mounted import SEAT_HORSE, mount
from .units import Unit

FLAME = P.hexrgb("#ff8a20")
GREEN_FLAME = P.hexrgb("#7cff6a")


def _flames(points, size, colour, seed=0) -> list:
    """Licks of fire rising from each point (for manes, tails, hooves)."""
    rng = np.random.default_rng(seed)
    out = []
    for k, p in enumerate(points):
        p = np.asarray(p, float)
        h = size * rng.uniform(0.8, 1.4)
        lean = rng.normal(0, 0.25, 2)
        tip = p + np.array([lean[0] * h, -0.6 * h + lean[1] * h, h])
        f = S.rod(p, tip, size * 0.42, S.mat(colour, glow=1.0), 0.05)
        f.casts = False
        f.ink = False
        out.append(f)
    return out


def _hair(base, seed=0) -> S.Material:
    """Mane and tail hair: dark, in strands (lighter and darker streaks along its length)."""
    base = np.asarray(base, float)

    def color(look):
        p = look.p
        streak = 0.5 + 0.5 * np.sin(p[:, 0] * 9.0 + P.noise(p, 1.2, seed) * 6.0)
        return P.mix(base * 1.5, base * 0.5, streak)
    return S.Material(color=color, shine=0.35, gloss=18)


def _joint(p, r, m) -> S.Shape:
    return S.ball(tuple(np.asarray(p, float)), r, m)


def _horse_leg(name, x, y, back, front: bool, skin, hoof_m, bony=False, camel=False, kind="", seed=0) -> Part:
    """A horse's leg: the forearm or gaskin, a knee (front) or a pointed hock (hind), a slim cannon, a fetlock, an
    angled pastern and a hoof (a camel's broad pad)."""
    s = -1.0 if not front else 1.0  # hind legs bend the other way
    lift = 1.25 if camel else 1.0
    hip = np.array([x, y, back - 4.0])
    if front:
        elbow = np.array([x, y - 0.5, back - 8.5])
        knee = np.array([x, y + 0.5, 6.2 * lift])
    else:
        elbow = np.array([x, y + 1.6, back - 8.8])  # the stifle, forward
        knee = np.array([x, y - 1.6, 6.6 * lift])   # the hock, pointing back
    fetlock = np.array([x, y + 0.2 * s, 2.0])
    pastern = np.array([x, y + 0.8 * s + (0.8 if not front else 0.0), 0.9])
    if bony:
        bone_m = skin
        shapes = [S.rod(tuple(hip), tuple(elbow), 0.75, bone_m, 0.6), _joint(elbow, 1.0, bone_m),
                  S.rod(tuple(elbow), tuple(knee), 0.6, bone_m, 0.5), _joint(knee, 0.9, bone_m),
                  S.rod(tuple(knee), tuple(fetlock), 0.45, bone_m, 0.42), _joint(fetlock, 0.7, bone_m),
                  S.rod(tuple(fetlock), tuple(pastern), 0.45, bone_m, 0.45)]
    else:
        upper = 1.55 if front else 1.75
        shapes = [S.rod(tuple(hip), tuple(elbow), upper * 1.2, skin, upper), _joint(elbow, upper * 0.95, skin),
                  S.rod(tuple(elbow), tuple(knee), upper * 0.9, skin, 0.75), _joint(knee, 0.95, skin),
                  S.rod(tuple(knee), tuple(fetlock), 0.68, skin, 0.62), _joint(fetlock, 0.82, skin),
                  S.rod(tuple(fetlock), tuple(pastern), 0.66, skin, 0.62)]
        if not front:  # the point of the hock
            shapes.append(S.Ellipsoid(skin, S._orient(tuple(knee + (0, -0.6, 0.3))), radii=(0.8, 0.9, 1.1)))
        if kind == "nightmare":  # long dark feathering round the fetlock
            shapes.append(S.rod(tuple(fetlock + (0, -0.2 * s, 1.2)), tuple(pastern + (0, -0.2 * s, -0.5)), 1.0,
                                _hair(P.hexrgb("#141016"), seed), 1.35))
    if camel:
        shapes.append(S.Ellipsoid(skin, S._orient(tuple(pastern + (0, 0.6, -0.4))), radii=(1.4, 1.8, 0.6)))
    else:
        shapes.append(S.rod(tuple(pastern + (0, 0.3 * s, 0.2)), tuple((x, pastern[1] + 0.5 * s, 0.0)), 0.95, hoof_m,
                            1.2))
    return Part(name, pivot=tuple(hip), boxes=shapes)


def _horse_head(kind, skin, eye, neck_top, back, seed, camel=False, bony=False) -> list:
    """The neck's top and the head: a broad forehead and a round jowl, the face tapering to a soft muzzle with
    nostrils and a mouth line, eyes set on the sides, pointed ears, a forelock."""
    P0 = np.asarray(neck_top, float)  # the poll
    M = P0 + (np.array([0, 5.2, -3.6]) if not camel else np.array([0, 5.0, -2.4]))  # the muzzle
    d = (M - P0) / np.linalg.norm(M - P0)
    dark = S.mat(P.hexrgb("#0e0a0c"))
    eye_m = S.mat(tuple(eye), glow=1.0)
    out = []
    if bony:  # a horse's skull: long, with deep sockets, a nose cavity and a row of teeth
        out += [S.Ellipsoid(skin, S._orient(tuple(P0 + d * 1.2 + (0, 0, 0.2))), radii=(1.7, 1.9, 1.6)),
                S.rod(tuple(P0 + d * 1.6), tuple(M), 1.25, skin, 0.85),
                S.Ellipsoid(skin, S._orient(tuple(P0 + (0, 1.0, -1.7))), radii=(1.1, 1.8, 1.3))]
        for sx in (1, -1):
            out += [S.Ellipsoid(dark, S._orient(tuple(P0 + d * 1.3 + (sx * 1.25, 0, 0.3))), radii=(0.5, 0.75, 0.6)),
                    S.ball(tuple(P0 + d * 1.3 + (sx * 1.55, 0.1, 0.3)), 0.24, eye_m)]
        out += [S.Ellipsoid(dark, S._orient(tuple(M + (0, 0.35, 0.45))), radii=(0.5, 0.8, 0.35)),
                S.Ellipsoid(S.mat(P.hexrgb("#d8ccb0")), S._orient(tuple(M + (0, 0.2, -0.85))), radii=(0.75, 0.9, 0.3))]
        return out
    out += [S.Ellipsoid(skin, S._orient(tuple(P0 + d * 1.1 + (0, 0, 0.3))), radii=(1.75, 1.7, 1.5)),  # forehead
            S.Ellipsoid(skin, S._orient(tuple(P0 + (0, 1.3, -1.6))), radii=(1.65, 1.9, 1.7)),  # the jowl
            S.rod(tuple(P0 + d * 1.6), tuple(M), 1.35, skin, 1.05),  # the face
            S.Ellipsoid(skin, S._orient(tuple(M)), radii=(1.25 if not camel else 1.4, 1.35, 1.1 if not camel else 1.3))]
    if camel:  # droopy lips
        out.append(S.Ellipsoid(skin, S._orient(tuple(M + (0, 0.5, -1.0))), radii=(1.1, 0.9, 0.6)))
    for sx in (1, -1):
        out += [S.Ellipsoid(dark, S._orient(tuple(M + (sx * 0.55, 0.95, 0.35))), radii=(0.32, 0.25, 0.42)),  # nostril
                S.Ellipsoid(dark, S._orient(tuple(P0 + d * 1.5 + (sx * 1.55, 0.0, 0.25))), radii=(0.32, 0.5, 0.36)),
                S.ball(tuple(P0 + d * 1.5 + (sx * 1.75, 0.05, 0.25)), 0.26, eye_m)]
        if camel:
            out.append(S.Ellipsoid(skin, S._orient(tuple(P0 + (sx * 1.3, -0.4, 1.0))), radii=(0.4, 0.6, 0.7)))
        else:  # pointed ears, cupped
            base = P0 + (sx * 0.85, -0.3, 1.1)
            out += [S.hull([base + (-0.5, -0.3, 0), base + (0.5, -0.3, 0), base + (0, 0.5, 0),
                            base + (sx * 0.35, 0.1, 2.6)], skin),
                    S.hull([base + (-0.25, 0.2, 0.3), base + (0.25, 0.2, 0.3), base + (sx * 0.3, 0.3, 2.0)], dark)]
    out.append(S.Ellipsoid(dark, S._orient(tuple(M + (0, 0.7, -0.75))), radii=(0.85, 0.7, 0.12)))  # the mouth line
    return out


def _bridle(neck_top, back, camel=False) -> list:
    """A bridle of dark leather: a noseband, a cheek strap, a brow band, and the reins back to the saddle."""
    leather = S.mat(P.hexrgb("#2a1a14"), shine=0.4)
    iron = P.metal(P.hexrgb("#4a3a2e"), rust=0.4)
    P0 = np.asarray(neck_top, float)
    M = P0 + (np.array([0, 5.2, -3.6]) if not camel else np.array([0, 5.0, -2.4]))
    nose = P0 + (M - P0) * 0.72
    out = [S.torus(tuple(nose), 1.35, 0.18, leather, rot=(-35, 0, 0)),
           S.torus(tuple(P0 + (0, 0.6, 0.4)), 1.75, 0.16, leather, rot=(-60, 0, 0))]
    for sx in (1, -1):
        out += [S.rod(tuple(P0 + (sx * 1.6, 0.4, 0.4)), tuple(nose + (sx * 1.25, -0.2, 0.2)), 0.16, leather),
                S.ball(tuple(M + (sx * 1.0, -0.1, -0.6)), 0.3, iron),
                S.rod(tuple(M + (sx * 1.0, -0.1, -0.6)), (sx * 1.8, 6.0, back + 2.4), 0.13, leather)]
    return out


def steed(kind: str, seed: int = 0, barding: Optional[str] = None, saddle: bool = True) -> Part:
    """A steed facing +y, its back at about z 17: a horse modelled muscle by muscle (barrel, deep chest, croup,
    withers, shoulders and thighs), an arched neck with a mane, a horse's head, jointed legs with hooves, a tail of
    hair; a saddle on a blanket in the owner's colour, a bridle and reins (the player found the old ones looked
    like balloons)."""
    if kind == "bonehorse":
        skin = P.bone(seed)
    elif kind == "ghosthorse":
        skin = replace(C.ghostly(P.hexrgb("#76a2e6"), seed), vivid=0.9)
    elif kind == "mummycamel":
        skin = C.bandage(seed=seed)
    else:  # nightmare
        skin = S.Material(color=lambda l: P.mix(P.hexrgb("#3e3640"), P.hexrgb("#120e12"),
                                                0.5 * P.fbm(l.p, 2.0, seed)), shine=0.35, gloss=20, vivid=0.3)
    camel = kind == "mummycamel"
    bony = kind == "bonehorse"
    eye = {"nightmare": FLAME, "bonehorse": P.hexrgb("#ff2410"), "ghosthorse": P.hexrgb("#ff2410"),
           "mummycamel": P.hexrgb("#ff2410")}[kind]
    hair_m = _hair(P.hexrgb("#141016") if kind == "nightmare" else P.hexrgb("#8aa8e8") if kind == "ghosthorse" else
                   P.hexrgb("#5a4a36"), seed)
    back = 17.0 if not camel else 20.0
    root = Part("root")
    body_shapes = []
    if bony:
        from .undead import _ribs_material
        body_shapes += [S.ball((0, 1.5, back - 2.8), (3.8, 6.8, 3.9), _ribs_material(seed)).clip((0, 0, -1), 1.0),
                        S.Ellipsoid(skin, S._orient((0, -6.5, back - 1.8)), radii=(3.4, 2.8, 2.0)),  # the pelvis
                        S.Ellipsoid(skin, S._orient((0, 5.6, back - 3.0), (25, 0, 0)), radii=(3.2, 1.0, 3.0))]  # scapulae
        body_shapes += [S.ball((0, y, back + 1.2 - abs(y) * 0.05), (0.75, 0.9, 1.0), skin) for y in np.linspace(-9, 8, 10)]
        body_shapes += [S.rod((0, y, back + 1.0), (0, y, back + 2.4), 0.3, skin, 0.15) for y in np.linspace(-6, 6, 6)]
    else:
        body_shapes += [S.Ellipsoid(skin, S._orient((0, 0.0, back - 3.3)), radii=(3.5, 7.4, 3.8)),  # the barrel
                        S.Ellipsoid(skin, S._orient((0, 6.2, back - 3.4)), radii=(3.6, 3.4, 4.3)),  # the chest
                        S.Ellipsoid(skin, S._orient((0, 8.6, back - 5.2)), radii=(2.9, 1.6, 2.8)),  # its front
                        S.Ellipsoid(skin, S._orient((0, 5.2, back + 0.1)), radii=(1.4, 2.8, 1.4)),  # the withers
                        S.Ellipsoid(skin, S._orient((0, -6.4, back - 2.6)), radii=(3.9, 4.3, 4.3)),  # hindquarters
                        S.Ellipsoid(skin, S._orient((0, -6.0, back - 0.2)), radii=(3.1, 3.6, 1.7))]  # the croup
        for sx in (1, -1):
            body_shapes += [S.Ellipsoid(skin, S._orient((sx * 2.4, 5.6, back - 4.6), (0, 0, 0)),
                                        radii=(1.6, 2.6, 3.6)),  # shoulder
                            S.Ellipsoid(skin, S._orient((sx * 2.6, -6.8, back - 5.0)), radii=(1.7, 3.0, 3.6))]  # thigh
        if camel:
            body_shapes += [S.Ellipsoid(skin, S._orient((0, -0.8, back + 1.6)), radii=(3.0, 4.2, 3.8))]  # the hump
    # saddle: a blanket in the owner's colour, a seat with a raised pommel and cantle, stirrups
    if saddle:
        sz = back + 0.9 + (3.4 if camel else 0.0)
        leather = S.mat(P.hexrgb("#3a2418"), shine=0.4)
        iron = P.metal(P.hexrgb("#3a2e26"), rust=0.4)
        body_shapes += [S.Ellipsoid(C.team_cloth(0.55, seed), S._orient((0, -1.2, sz - 0.6)), radii=(4.1, 3.8, 1.3))
                        .clip((0, 0, -1), -(sz - 1.2)),
                        S.Ellipsoid(leather, S._orient((0, -1.2, sz + 0.3)), radii=(2.6, 3.0, 0.8)),
                        S.Ellipsoid(leather, S._orient((0, 1.6, sz + 0.9)), radii=(1.6, 0.8, 1.0)),
                        S.Ellipsoid(leather, S._orient((0, -3.9, sz + 1.1)), radii=(2.0, 0.8, 1.2))]
        for sx in (1, -1):
            body_shapes += [S.rod((sx * 3.0, -0.8, sz), (sx * 3.6, -0.6, sz - 6.0), 0.18, leather),
                            S.torus((sx * 3.6, -0.6, sz - 6.6), 0.7, 0.18, iron, rot=(0, 90, 0))]
    if barding == "plate":  # a peytral over the chest, a crinet down the neck (below), riveted dark plate
        plate = P.metal(C.KNIGHT_PLATE, rust=0.5, seed=seed, shine=0.15)
        body_shapes += [S.Ellipsoid(plate, S._orient((0, 6.4, back - 3.8)), radii=(3.95, 3.7, 4.0)).clip((0, -1, 0), -6.4),
                        S.Ellipsoid(plate, S._orient((0, -6.4, back - 2.4)), radii=(4.2, 4.6, 4.1))
                        .clip((0, 1, 0), -6.6).clip((0, 0, -1), -(back - 6.0))]
    elif barding == "cloth":  # a caparison in the owner's colour, hanging in folds, its hem torn
        def stripes(look):
            return np.where((look.p[:, 1] % 4.0) < 0.6, 0.4, 0.75)
        body_shapes += [S.Ellipsoid(P.team(0.7, stripes), S._orient((0, 0.0, back - 3.2)), radii=(4.0, 10.4, 4.3))
                        .clip((0, 0, 1), back - 0.8).clip((0, 0, -1), -(back - 8.4))]
    body_part = Part("horse_body", pivot=(0, 0, back - 1.0), boxes=body_shapes)

    # neck (arched, a crest on top) and head
    base = np.array([0, 7.6, back - 1.8])
    neck_top = np.array([0, 13.0, back + 7.2]) if not camel else np.array([0, 15.0, back + 6.0])
    head_shapes = []
    if bony:
        head_shapes += [S.ball(tuple(base + (neck_top - base) * f), 1.0 - 0.15 * f, skin) for f in np.linspace(0, 1, 7)]
    elif camel:
        mid = np.array([0, 12.8, back - 2.6])
        head_shapes += S.limb(tuple(base), tuple(mid), 2.2, skin, 1.7) + S.limb(tuple(mid), tuple(neck_top), 1.7, skin,
                                                                               1.5)
    else:
        crest = base + (neck_top - base) * 0.5 + (0, -0.9, 1.0)
        head_shapes += [S.rod(tuple(base), tuple(neck_top), 2.9, skin, 1.55),
                        S.Ellipsoid(skin, S.axis_frame(base + (0, -0.6, 1.6), neck_top + (0, -0.6, 0.4)) @
                                    S.affine(None, (0, 0, float(np.linalg.norm(neck_top - base)) / 2)),
                                    radii=(1.5, 1.6, float(np.linalg.norm(neck_top - base)) / 2 + 0.6)),
                        S.Ellipsoid(skin, S._orient(tuple(base + (0, 1.6, -1.0))), radii=(2.6, 2.0, 2.6))]
        # the mane: locks of hair falling to one side of the crest, a forelock between the ears
        for f in np.linspace(0.05, 1.0, 9):
            p = base + (0, -0.6, 2.2) + (neck_top + (0, -0.8, 1.0) - (base + (0, -0.6, 2.2))) * f
            head_shapes.append(S.rod(tuple(p + (0.2, 0, 0.4)), tuple(p + (1.6, -0.3, -2.4)), 0.75, hair_m, 0.2))
        head_shapes.append(S.rod(tuple(neck_top + (0, 0.6, 1.2)), tuple(neck_top + (0.3, 2.2, -0.6)), 0.55, hair_m, 0.15))
    head_shapes += _horse_head(kind, skin, eye, neck_top, back, seed, camel, bony)
    if saddle and not bony:
        head_shapes += _bridle(neck_top, back, camel)
    if kind == "nightmare":  # fire licking up from the mane
        pts = [base + (0, -0.6, 2.4) + (neck_top + (0, -0.8, 1.4) - (base + (0, -0.6, 2.4))) * f
               for f in np.linspace(0.1, 1, 5)]
        head_shapes += _flames(pts, 1.8, FLAME, seed)
    if barding == "plate":  # a crinet down the neck, a chanfron over the face with a spike
        plate = P.metal(C.KNIGHT_PLATE, rust=0.5, seed=seed, shine=0.15)
        M = neck_top + np.array([0, 5.2, -3.6])
        head_shapes += [S.rod(tuple(base + (0, -0.4, 1.0)), tuple(neck_top + (0, -0.6, 0.6)), 2.4, plate, 1.4)
                        .clip((0, 1, 0), 0.0),
                        S.rod(tuple(neck_top + (0, 0.8, 0.9)), tuple(M + (0, 0.0, 0.6)), 1.6, plate, 1.25)
                        .clip((0, 0, -1), -(neck_top[2] - 3.4)),
                        S.rod(tuple(neck_top + (0, 1.4, 1.4)), tuple(neck_top + (0, 2.4, 4.6)), 0.55, plate, 0.05)]
    head_part = Part("horse_head", pivot=(0, 8.0, back), boxes=head_shapes)

    # tail: locks of hair flowing from the dock (bones on the skeleton, fire on the nightmare)
    t0 = np.array([0, -10.4, back - 0.6])
    if bony:
        tail = [S.ball(tuple(t0 + (0, -k * 1.1, -k * 1.5)), 0.7 - 0.08 * k, skin) for k in range(6)]
    else:
        tail = [S.rod(tuple(t0), tuple(t0 + (0, -1.6, -1.2)), 1.0, skin, 0.8)]
        rng = np.random.default_rng(seed + 3)
        for j in range(7):
            spread = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-0.8, 0.6), 0.0])
            a = t0 + (0, -1.4, -0.8)
            b = a + (0, -2.6, -4.0) + spread * 0.8
            c = b + (0, -0.8, -4.6) + spread * 1.4
            tail += [S.rod(tuple(a), tuple(b), 0.85, hair_m, 0.6), S.rod(tuple(b), tuple(c), 0.6, hair_m, 0.12)]
        if kind == "nightmare":
            tail += _flames([t0 + (0, -2.0 - k * 0.8, -2.0 - k * 1.6) for k in range(4)], 1.8, FLAME, seed + 1)
    tail_part = Part("tail", pivot=tuple(t0), boxes=tail)
    body_part.add(head_part, tail_part)

    hoof_m = S.mat(P.hexrgb("#141012"), shine=0.4) if kind != "ghosthorse" else skin
    fx, bx = 2.3, 2.5
    legs = [_horse_leg("leg_fr", fx, 6.5, back, True, skin, hoof_m, bony, camel, kind, seed),
            _horse_leg("leg_fl", -fx, 6.5, back, True, skin, hoof_m, bony, camel, kind, seed),
            _horse_leg("leg_br", bx, -6.5, back, False, skin, hoof_m, bony, camel, kind, seed),
            _horse_leg("leg_bl", -bx, -6.5, back, False, skin, hoof_m, bony, camel, kind, seed)]
    if kind == "nightmare":
        for lg in legs:
            x, y = lg.pivot[0], lg.pivot[1]
            lg.boxes += _flames([(x, y + 0.4, 0.3)], 1.3, FLAME, seed + int(x * 10 + y))
    root.add(body_part, *legs)
    todo = [root]
    while todo:  # the barrel, chest, croup, shoulders, thighs, neck and head melt into one horse (sculpt.BLEND)
        part = todo.pop()
        todo += part.children
        A.soft([s for s in part.boxes if isinstance(s, S.Shape) and s.material is skin], "bone" if bony else "flesh")
    if kind == "ghosthorse":
        root.offset = (0.0, 0.0, 1.5)
    return root


SEAT = SEAT_HORSE + 2.0


def _rider(key, kind, gear, seed, attack="chop", **bkw) -> Unit:
    return person(key + "_rider", "", "", body(kind, seed, **bkw), gear, attack=attack, seed=seed)


def horseman(key, name, replaces, rider_kind, gear, steed_kind, attack="chop", barding=None, civ=None, seed=0,
             camel=False) -> Unit:
    rider = _rider(key, rider_kind, gear, seed, attack)
    u = mount(key, name, replaces, rider, steed(steed_kind, seed + 50, barding), SEAT + (3.0 if camel else 0.0),
              civ=civ)
    return u


def headless_horseman(key="paladin") -> Unit:
    """The Paladin: the Headless Horseman on a nightmare, a jack-o'-lantern held high in his left hand, a sword in
    his right, a long cape in the owner's colour."""
    rider = person(key + "_rider", "", "", body("knight", 61), Gear(armour="plate", cape=True, right="longsword"),
                   attack="chop", seed=61)
    rider.root.find("head").boxes = [S.rod((0, -0.2, 22.5), (0, 0.1, 24.0), 1.6, P.metal(P.hexrgb("#2e2a32")),
                                           1.4)]
    arm_l = rider.root.find("arm_l")
    hand = np.array([-4.4, 0.9, 12.6])
    lantern = PK.lantern(hand + (0, 0.6, -0.4), 2.4, PK.SCARY, seed=4)
    arm_l.rot = (150, 0, 0)  # held up high
    arm_l.add(Part("item_l", pivot=tuple(hand), boxes=lantern))
    return mount(key, "Headless Horseman", "Paladin", rider, steed("nightmare", 62, "plate"), SEAT)


MOUNTED = {
    "scout_cavalry": lambda: horseman("scout_cavalry", "Zombie Rider", "Scout Cavalry", "zombie",
                                      Gear(cloth="rags", right="sword"), "bonehorse", seed=1),
    "light_cavalry": lambda: horseman("light_cavalry", "Skeleton Rider", "Light Cavalry", "skeleton",
                                      Gear(helmet="hood", cloth="tunic", right="sword"), "bonehorse", seed=2),
    "hussar": lambda: horseman("hussar", "Wraith Hussar", "Hussar", "ghost",
                               Gear(helmet="kettle", cloth="tunic", cape=True, right="longsword"), "ghosthorse",
                               seed=3),
    "knight": lambda: horseman("knight", "Death Knight Rider", "Knight", "knight",
                               Gear(armour="plate", right="longsword", shield="kite"), "nightmare", seed=4),
    "cavalier": lambda: horseman("cavalier", "Dread Cavalier", "Cavalier", "knight",
                                 Gear(armour="plate", cape=True, right="longsword", shield="kite"), "nightmare",
                                 barding="cloth", seed=5),
    "paladin": headless_horseman,
    "cavalry_archer": lambda: horseman("cavalry_archer", "Skeleton Horse Archer", "Cavalry Archer", "skeleton",
                                       Gear(helmet="hood", cloth="tunic", left="bonebow", back="quiver"),
                                       "bonehorse", attack="bow", seed=6),
    "heavy_cavalry_archer": lambda: horseman("heavy_cavalry_archer", "Wraith Horse Archer", "Heavy Cavalry Archer",
                                             "ghost", Gear(helmet="hood", armour="mail", left="bow", back="quiver"),
                                             "ghosthorse", attack="bow", seed=7),
    "camel": lambda: horseman("camel", "Mummy Camel Rider", "Camel", "mummy",
                              Gear(helmet="turban", cloth="tunic", right="spear"), "mummycamel", seed=8, camel=True),
    "heavy_camel": lambda: horseman("heavy_camel", "Mummy Lancer", "Heavy Camel", "mummy",
                                    Gear(helmet="turban", armour="mail", cape=True, right="spear"), "mummycamel",
                                    barding="cloth", seed=9, camel=True),
    "cataphract": lambda: horseman("cataphract", "Revenant Cataphract", "Cataphract", "draugr",
                                   Gear(helmet="great", armour="plate", cape=True, right="spear"), "nightmare",
                                   barding="plate", civ="Byzantines", seed=10),
    "mangudai": lambda: horseman("mangudai", "Ghost Mangudai", "Mangudai", "ghost",
                                 Gear(helmet="pelt", armour="lamellar", left="bow", back="quiver"), "ghosthorse",
                                 attack="bow", civ="Mongols", seed=11),
    "mameluke": lambda: horseman("mameluke", "Mummy Mameluke", "Mameluke", "mummy",
                                 Gear(helmet="turban", cloth="tunic", right="sword"), "mummycamel", attack="throw",
                                 civ="Saracens", seed=12, camel=True),
    "tarkan": lambda: horseman("tarkan", "Ghoul Raider", "Tarkan", "ghoul",
                               Gear(cloth="rags", right="torch", shield="round"), "nightmare", civ="Huns", seed=13),
    "conquistador": lambda: horseman("conquistador", "Phantom Conquistador", "Conquistador", "skeleton",
                                     Gear(helmet="kettle", armour="plate", chest="gun"), "nightmare", attack="gun",
                                     civ="Spanish", seed=14),
    "missionary": lambda: horseman("missionary", "Grave Priest", "Missionary", "skeleton",
                                   Gear(helmet="cowl", cloth="robe", right="staff"), "bonehorse", attack="cast",
                                   civ="Spanish", seed=15),
}
