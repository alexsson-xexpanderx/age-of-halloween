"""Mounted units: undead riders on undead steeds.

Steeds use the horse rig's parts (animation.py "cavalry" / "horse_archer"): "horse_body" (with the rider, "horse_head"
and "tail" on it) and "leg_fr", "leg_fl", "leg_br", "leg_bl" pivoting at the hips and shoulders. Riders are
creatures.py people seated as Age of Minecraft seated its riders (mounted.mount).

    nightmare   a black horse, flaming mane and tail, burning eyes and hooves
    bonehorse   a horse's skeleton
    ghosthorse  a pale spectral horse, glowing
    mummycamel  a camel wrapped in bandages
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

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


def steed(kind: str, seed: int = 0, barding: Optional[str] = None) -> Part:
    """A steed facing +y, its back at about z 16."""
    if kind == "bonehorse":
        skin = P.bone(seed)
    elif kind == "ghosthorse":
        skin = C.ghostly(P.hexrgb("#a8d8d0"), seed)
    elif kind == "mummycamel":
        skin = C.bandage(seed=seed)
    else:  # nightmare
        skin = S.Material(color=lambda l: P.mix(P.hexrgb("#24202a"), P.hexrgb("#08060a"),
                                                0.4 * P.fbm(l.p, 2.0, seed)), shine=0.45, gloss=24)
    camel = kind == "mummycamel"
    bony = kind == "bonehorse"
    eye = {"nightmare": FLAME, "bonehorse": P.hexrgb("#9cff6a"), "ghosthorse": P.hexrgb("#e0fffa"),
           "mummycamel": P.hexrgb("#9cff6a")}[kind]
    back = 17.0 if not camel else 20.0
    root = Part("root")
    body_shapes = []
    if bony:
        from .undead import _ribs_material
        body_shapes += [S.ball((0, 1.5, back - 2.5), (4.4, 7.5, 4.2), _ribs_material(seed)).clip((0, 0, -1), 1.0),
                        S.ball((0, -6.5, back - 1.5), (4.0, 3.2, 3.2), skin)]
        body_shapes += [S.ball((0, y, back + 1.5), (0.9, 0.9, 0.8), skin) for y in np.linspace(-8, 8, 8)]
    else:
        body_shapes += [S.ball((0, 0.0, back - 3.0), (4.6, 10.0, 4.8), skin),
                        S.ball((0, 6.5, back - 2.4), (4.4, 4.6, 4.8), skin),
                        S.ball((0, -6.5, back - 2.2), (4.6, 4.8, 4.8), skin)]
        if camel:
            body_shapes += [S.ball((0, -1.0, back + 1.6), (3.2, 4.0, 3.8), skin)]
    # saddle and blanket in the owner's colour
    body_shapes += [S.ball((0, -1.5, back + 1.4 + (2.6 if camel else 0)), (4.0, 3.6, 1.2), C.team_cloth(0.55, seed))
                    .clip((0, 0, -1), -(back + 0.8 + (2.6 if camel else 0)))]
    if barding == "plate":
        plate = P.metal(P.hexrgb("#3a363e"), rust=0.4, seed=seed, shine=0.7)
        body_shapes += [S.ball((0, 5.0, back - 2.0), (4.9, 4.4, 4.0), plate).clip((0, 0, 1), back + 0.5)]
    elif barding == "cloth":
        def stripes(look):
            return np.where((look.p[:, 1] % 4.0) < 0.6, 0.4, 0.75)
        body_shapes += [S.ball((0, 0.0, back - 3.0), (4.9, 10.3, 5.0), P.team(0.7, stripes))
                        .clip((0, 0, 1), back - 1.5).clip((0, 0, -1), -(back - 7.0))]
    body_part = Part("horse_body", pivot=(0, 0, back - 1.0), boxes=body_shapes)

    # neck and head
    neck_top = np.array([0, 13.0, back + 7.5]) if not camel else np.array([0, 14.0, back + 6.0])
    neck = (S.limb((0, 8.0, back - 1.0), tuple(neck_top), 2.6, skin, 1.9) if not bony else
            [S.ball(tuple(np.array([0, 8.0, back - 1.0]) + (neck_top - np.array([0, 8.0, back - 1.0])) * f),
                    1.1, skin) for f in np.linspace(0, 1, 6)])
    head_shapes = list(neck)
    if camel:
        mid = np.array([0, 12.5, back - 2.0])
        head_shapes = S.limb((0, 8.0, back - 1.0), tuple(mid), 2.2, skin, 1.8) + \
            S.limb(tuple(mid), (0, 14.5, back + 6.0), 1.8, skin, 1.6)
    hx = neck_top + np.array([0, 2.6, -1.4])
    if bony:
        skull_m = C.eyes_material(skin, [(-1.1, 0.6, 0.6, 0.5), (1.1, 0.6, 0.6, 0.5)], eye, 0.55)
        head_shapes += [S.Ellipsoid(skull_m, S.axis_frame(neck_top, hx + (0, 4.0, -2.0)) @
                                    S.affine(None, (0, 0, 2.6)), radii=(1.8, 1.9, 3.6))]
    else:
        head_m = C.eyes_material(skin, [(-1.5, 0.6, 0.55, 0.5), (1.5, 0.6, 0.55, 0.5)], eye, 0.6)
        head_shapes += [S.Ellipsoid(head_m, S.axis_frame(neck_top, hx + (0, 4.2, -2.6)) @
                                    S.affine(None, (0, 0, 2.8)), radii=(2.0, 2.2, 3.8))]
        for sx in (-1, 1):  # ears
            head_shapes.append(S.rod(neck_top + (sx * 1.0, 0.6, 1.0), neck_top + (sx * 1.4, 0.0, 3.4), 0.6,
                                     skin, 0.1))
    if kind == "nightmare":  # a mane of fire down the neck
        pts = [np.array([0, 8.0, back - 0.4]) + (neck_top + (0, -0.6, 1.2) - np.array([0, 8.0, back - 0.4])) * f
               for f in np.linspace(0, 1, 6)]
        head_shapes += _flames(pts, 2.6, FLAME, seed)
    if barding == "plate":  # a chanfron on the face with a spike
        plate = P.metal(P.hexrgb("#3a363e"), rust=0.4, seed=seed, shine=0.7)
        head_shapes += [S.Ellipsoid(plate, S.axis_frame(neck_top, hx + (0, 4.2, -2.6)) @
                                    S.affine(None, (0, 0.4, 3.2)), radii=(2.1, 1.8, 2.8)).clip((0, -1, 0), -0.6),
                        S.rod(neck_top + (0, 1.8, 1.0), neck_top + (0, 2.6, 4.6), 0.6, plate, 0.05)]
    head_part = Part("horse_head", pivot=(0, 8.0, back), boxes=head_shapes)

    # tail
    t0 = np.array([0, -10.5, back - 0.5])
    if kind == "nightmare":
        tail = _flames([t0 + (0, -k * 1.0, -k * 1.4) for k in range(5)], 2.4, FLAME, seed + 1)
    elif bony:
        tail = [S.ball(tuple(t0 + (0, -k * 1.1, -k * 1.5)), 0.7 - 0.08 * k, skin) for k in range(6)]
    else:
        tail = S.limb(tuple(t0), tuple(t0 + (0, -2.5, -8.0)), 1.2, skin, 0.4)
    tail_part = Part("tail", pivot=tuple(t0), boxes=tail)
    body_part.add(head_part, tail_part)

    def leg(name, x, y):
        hip = (x, y, back - 4.0)
        knee = (x, y + (0.8 if y > 0 else -0.8), back * 0.42)
        hoof = (x, y, 1.0)
        r = 1.3 if not bony else 0.8
        shapes = C._limb_shapes(body("skeleton" if bony else "zombie", seed, skin=skin), hip, knee, r, r * 0.8) \
            if bony else S.limb(hip, knee, r * 1.25, skin, r) + S.limb(knee, hoof, r * 0.85, skin, r * 0.7)
        if bony:
            shapes += C._limb_shapes(body("skeleton", seed, skin=skin), knee, hoof, r * 0.8, r * 0.7)
        hoof_m = S.mat(P.hexrgb("#141012")) if kind != "ghosthorse" else skin
        shapes.append(S.rod((x, y, 0.0), (x, y, 1.4), 1.2, hoof_m, 1.0))
        if kind == "nightmare":
            shapes += _flames([(x, y, 0.2)], 1.6, FLAME, seed + int(x * 10 + y))
        return Part(name, pivot=hip, boxes=shapes)

    fx, bx = 2.6, 2.6
    root.add(body_part, leg("leg_fr", fx, 6.5), leg("leg_fl", -fx, 6.5), leg("leg_br", bx, -6.5),
             leg("leg_bl", -bx, -6.5))
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
