"""Siege engines and the ghost train, on the old rigs' parts (animation.py):

    ravager    "torso", "head" (rams forward), "leg_fr/fl/br/bl"   -> coffin rams on skeleton legs
    wheeled    "body", "weapon" (kicks when it fires), "wheel_1".., "flag", "smoke_1/2", a "draught" animal
    trebuchet  "base", "frame", "arm", "counterweight", "sling", "wheel_1".., "flag"

Sizes follow Age of Minecraft's models (the same `scale`), so they fill the same room on the map.
"""
from __future__ import annotations

import math

import numpy as np

from . import cavalry as CV
from . import creatures as C
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part
from .units import Unit

WOOD = lambda s=0: P.wood(P.hexrgb("#3e2e26"), P.hexrgb("#1a120e"), along=1, seed=s)  # noqa: E731
WOOD_UP = lambda s=0: P.wood(P.hexrgb("#3e2e26"), P.hexrgb("#1a120e"), along=2, seed=s)  # noqa: E731
IRON = lambda s=0: P.metal(P.hexrgb("#3a363e"), rust=0.6, seed=s)  # noqa: E731


def wheel(name, at, r, seed=0, spokes=8, bone=False) -> Part:
    """A wheel in the y-z plane, rolling forward along +y: an iron rim, a hub and spokes (bones, if `bone`)."""
    at = np.asarray(at, float)
    rim_m = IRON(seed)
    spoke_m = P.bone(seed) if bone else WOOD(seed)
    shapes = [S.torus(tuple(at), r - 0.45, 0.55, rim_m, rot=(0, 90, 0)),
              S.rod(tuple(at - (0.9, 0, 0)), tuple(at + (0.9, 0, 0)), r * 0.22, rim_m)]
    for k in range(spokes):
        a = 2 * math.pi * k / spokes
        tip = at + np.array([0, math.cos(a), math.sin(a)]) * (r - 0.6)
        shapes.append(S.rod(tuple(at), tuple(tip), 0.3 if not bone else 0.35, spoke_m))
    return Part(name, pivot=tuple(at), boxes=shapes)


def team_flag(at, height=12, seed=0) -> Part:
    """A tattered pennant in the owner's colour on a pole."""
    at = np.asarray(at, float)
    pole = S.rod(tuple(at), tuple(at + (0, 0, height)), 0.3, WOOD_UP(seed))
    top = at + (0, 0, height)

    def torn(look):
        p = look.p
        return np.where(np.sin(p[:, 1] * 3.0 + p[:, 2] * 2.0) > 0.85, 0.3, 0.7)
    cloth = S.hull([top - (0, 0, 0.3), top - (0, 0, 4.0), top + (0, -6.5, -1.4), top + (0, -5.5, -2.8),
                    top + (0.15, -3, -2.0), top + (-0.15, -3, -2.0)], P.team(0.65, torn))
    return Part("flag", boxes=[pole, cloth])


def _siege(key, name, replaces, root, attack="fire", rig="wheeled", scale=1.5, civ=None) -> Unit:
    return Unit(key, name, replaces, rig, root, group="siege", attack=attack, civ=civ, scale=scale)


# --------------------------------------------------------------------------- rams

def coffin_ram(key, name, replaces, level=0) -> Unit:
    """A great coffin carried on four skeleton legs, an iron-capped log with a skull on its end for the ram;
    the capped and siege rams get iron bands, then spikes and a team-coloured shroud."""
    seed = 40 + level
    wood = WOOD(seed)
    lid_z = 22.0
    w, l = 9.0, 20.0
    # the coffin: six-sided, widest at the shoulders, a peaked lid
    def coffin_pts(z0, z1, grow=0.0):
        prof = [(-w * 0.7 - grow, -l - grow), (w * 0.7 + grow, -l - grow), (w + grow, l * 0.25),
                (w * 0.8 + grow, l + grow), (-w * 0.8 - grow, l + grow), (-w - grow, l * 0.25)]
        return [(x, y, z) for x, y in prof for z in (z0, z1)]
    body = [S.hull(coffin_pts(10.0, lid_z), wood),
            S.hull(coffin_pts(lid_z, lid_z + 1.6, 0.6) + [(0, y, lid_z + 4.5) for y in (-l * 0.9, l * 0.9)],
                   WOOD(seed + 1))]
    # a cross on the lid, iron bands, studs
    body.append(S.block((-0.7, -l * 0.6, lid_z + 3.6), (0.7, l * 0.5, lid_z + 4.9), IRON(seed)))
    body.append(S.block((-4.0, l * 0.15, lid_z + 3.4), (4.0, l * 0.3, lid_z + 4.7), IRON(seed)))
    if level >= 1:
        for y in (-l * 0.6, 0.0, l * 0.6):
            body.append(S.hull([(x, y + d, z) for x in (-w - 0.4, w + 0.4) for d in (-0.6, 0.6)
                                for z in (9.6, lid_z + 1.8)], IRON(seed)))
    if level >= 2:
        for x in (-w - 0.6, w + 0.6):
            for y in (-l * 0.5, l * 0.1, l * 0.6):
                body.append(S.rod((x, y, 16.0), (x * 1.25, y, 16.5), 0.8, IRON(seed), 0.05))
        body.append(S.hull(coffin_pts(lid_z + 1.6, lid_z + 2.2, 0.8), C.team_cloth(0.5, seed)))
    else:
        body.append(S.block((-w * 0.6, -l * 0.3, lid_z + 1.4), (w * 0.6, l * 0.1, lid_z + 2.4),
                            C.team_cloth(0.55, seed)))
    torso = Part("torso", pivot=(0, 0, 16.0), boxes=body)
    # the ram: a log with an iron cap and a skull
    log = [S.rod((0, -6.0, 15.0), (0, l + 4.0, 15.0), 2.4, P.bark(seed=seed)),
           S.rod((0, l + 3.0, 15.0), (0, l + 6.5, 15.0), 2.8, IRON(seed), 2.4)]
    from .graveyard import skull
    sk = skull((0, 0, 0), 2.2, seed)
    for s in sk:
        s.xf = S.affine(S.rotation(0, 0, 0), (0, l + 7.0, 12.8)) @ s.xf
    head = Part("head", pivot=(0, 0, 15.0), boxes=log + sk)
    sk_body = C.body("skeleton", seed)

    def leg(name, x, y):
        knee = (x * 1.1, y + 1.5, 6.0)
        shapes = C._limb_shapes(sk_body, (x, y, 10.0), knee, 1.0, 0.8) + C._limb_shapes(sk_body, knee, (x, y, 1.2),
                                                                                       0.8, 0.7)
        shapes.append(S.ball((x, y + 1.0, 0.7), (1.0, 1.8, 0.7), sk_body.skin))
        return Part(name, pivot=(x, y, 10.0), boxes=shapes)
    root = Part("root").add(torso.add(head), leg("leg_fr", 6.5, 12.0), leg("leg_fl", -6.5, 12.0),
                            leg("leg_br", 6.0, -13.0), leg("leg_bl", -6.0, -13.0))
    return _siege(key, name, replaces, root, attack="ram", rig="ravager", scale=1.1)


# --------------------------------------------------------------------------- catapults and ballistae

def _cart(seed, length=24.0, width=14.0, wheels=4, r=4.0, bone_wheels=True):
    y0, y1 = -length * 0.6, length * 0.4
    wood = WOOD(seed)
    body = [S.block((-width / 2, y0, r * 0.9), (width / 2, y1, r * 0.9 + 2.0), wood)]
    for x in (-width / 2 - 0.4, width / 2 + 0.4):
        body.append(S.block((x - 0.6, y0, r * 0.9 - 0.6), (x + 0.6, y1, r * 0.9 + 2.6), WOOD(seed + 1)))
    ys = [y0 + 3, y1 - 3] if wheels == 4 else [(y0 + y1) / 2]
    parts = []
    k = 1
    for y in ys:
        for x in (width / 2 + 1.6, -width / 2 - 1.6):
            parts.append(wheel(f"wheel_{k}", (x, y, r), r, seed + k, bone=bone_wheels))
            k += 1
    return body, parts


def bone_catapult(key, name, replaces, level=0) -> Unit:
    """A catapult of bones and black timber: its throwing arm a giant's thigh bone ending in a ribcage cup, a
    flaming skull in it; skulls stacked on the frame."""
    seed = 50 + level
    body, wheels = _cart(seed)
    bone = P.bone(seed)
    wood = WOOD_UP(seed)
    for x in (-5.5, 5.5):  # the uprights and the cross bar the arm hits
        body.append(S.rod((x, -2.0, 6.0), (x, -2.0, 18.0), 0.9, wood))
    body.append(S.rod((-6.2, -2.0, 18.0), (6.2, -2.0, 18.0), 0.9, wood))
    body.append(S.rod((-6.0, -9.0, 7.0), (6.0, -9.0, 7.0), 1.6, P.bark(seed=seed)))  # the winding drum
    from .graveyard import skull
    for k, (x, y) in enumerate(((-4.0, 6.0), (-1.5, 7.0), (-3.0, 3.6))):  # spare skulls on the cart
        body += [s for s in skull((x, y, 7.0), 1.3, seed + k)]
    if level >= 1:  # iron bands and a team pennant
        for y in (-10.0, 4.0):
            body.append(S.block((-7.4, y - 0.5, 3.0), (7.4, y + 0.5, 7.2), IRON(seed)))
    arm_pts = [np.array([0, -9.0, 7.5]), np.array([0, 2.0, 16.0])]
    arm = [S.rod(tuple(arm_pts[0]), tuple(arm_pts[1]), 1.1, bone, 0.8), S.ball(tuple(arm_pts[0]), 1.9, bone),
           S.ball(tuple(arm_pts[1]), 1.6, bone)]
    cup_c = arm_pts[1] + (0, 1.4, 1.6)
    from .undead import _ribs_material
    arm.append(S.ball(tuple(cup_c), (2.4, 2.4, 1.8), _ribs_material(seed)).clip((0, 0, 1), 0.4))
    fire_skull = skull(tuple(cup_c + (0, 0, 0.2)), 1.4, seed + 9)
    flame = S.Ellipsoid(S.mat(P.SLIME if level >= 2 else P.hexrgb("#ff8a20"), glow=1.0),
                        S._orient(cup_c + (0, -0.6, 4.2)), radii=(1.6, 1.6, 2.8))
    flame.casts = False
    weapon = Part("weapon", pivot=tuple(arm_pts[0]), rot=(-25, 0, 0), boxes=arm + fire_skull + [flame])
    root = Part("root").add(Part("body", boxes=body).add(weapon), *wheels, team_flag((6.0, -12.0, 6.0), 12, seed))
    titles = ["Bone Catapult", "Bone Onager", "Siege Bone Onager"]
    return _siege(key, titles[level] if name is None else name, replaces, root, scale=1.6)


def bone_ballista(key, name, replaces, heavy=False) -> Unit:
    """A ballista whose bow is a pair of curved ribs and whose bolts are sharpened bones."""
    seed = 60 + heavy
    body, wheels = _cart(seed, 22.0, 12.0)
    bone = P.bone(seed)
    body.append(S.rod((0, -3.0, 6.0), (0, -3.0, 11.0), 1.4, WOOD_UP(seed)))
    stock = [S.block((-1.0, -9.0, 11.0), (1.0, 7.0, 12.6), WOOD(seed))]
    for sx in (1, -1):  # the bow: curved ribs
        pts = [np.array([0, 5.0, 11.8]) + np.array([sx * 3.0 * k, -1.2 * k * k, 0.2 * k]) for k in range(4)]
        for a, b in zip(pts, pts[1:]):
            stock.append(S.rod(tuple(a), tuple(b), 0.75 - 0.12 * np.linalg.norm(b - pts[0]) / 3, bone))
        stock.append(S.rod(tuple(pts[-1]), (0, -6.0, 12.2), 0.08, S.mat(P.hexrgb("#c8c0a8"))))
    bolt = S.rod((0, -5.5, 12.9), (0, 9.0, 12.9), 0.4, bone, 0.05)
    stock.append(bolt)
    if heavy:
        stock.append(S.rod((1.6, -5.5, 12.9), (1.6, 8.5, 12.9), 0.4, bone, 0.05))
        stock.append(S.rod((-1.6, -5.5, 12.9), (-1.6, 8.5, 12.9), 0.4, bone, 0.05))
        from .graveyard import skull
        stock += skull((0, 6.6, 12.6), 1.0, seed)
    weapon = Part("weapon", pivot=(0, -3.0, 11.5), boxes=stock)
    thrower = Part("bolt_thrower")
    root = Part("root").add(Part("body", boxes=body).add(weapon, thrower), *wheels,
                            team_flag((-5.0, -10.0, 6.0), 10, seed))
    return _siege(key, name, replaces, root, scale=1.5)


def gargoyle_bombard(key="bombard_cannon") -> Unit:
    """A heavy black cannon whose muzzle is a gargoyle's open mouth, glowing green inside."""
    seed = 70
    body = [S.block((-4.0, -16.0, 3.4), (4.0, 6.0, 7.0), WOOD(seed)),
            S.block((-1.2, -18.0, 0.0), (1.2, -12.0, 4.0), WOOD(seed + 1))]
    iron = P.metal(P.hexrgb("#2a282e"), rust=0.4, seed=seed, shine=0.7)
    barrel = [S.rod((0, -10.0, 9.5), (0, 12.0, 10.5), 3.0, iron, 2.5),
              S.ball((0, -10.0, 9.5), 3.2, iron)]
    for y in (-6.0, 2.0, 8.0):
        barrel.append(S.torus((0, y, 9.7 + y * 0.045), 2.8 - y * 0.02, 0.5, iron, rot=(90, 0, 0)))
    head_c = np.array([0, 13.5, 10.6])
    mouth = S.mat(P.SLIME, glow=1.0)
    gm = C.eyes_material(P.mottled(P.hexrgb("#4a4650"), P.hexrgb("#24222a"), 1.5, 0.4, seed), [(-1.4, 1.2, 0.6, 0.5),
                                                                                               (1.4, 1.2, 0.6, 0.5)],
                         P.hexrgb("#ff2410"), 0.6)
    barrel += [S.Ellipsoid(gm, S._orient(head_c), radii=(3.6, 3.4, 3.4)).clip((0, 1, -0.2), 1.6),
               S.ball(tuple(head_c + (0, 1.0, -0.4)), (2.0, 1.0, 1.6), mouth),
               S.rod(tuple(head_c + (-2.4, -0.6, 2.4)), tuple(head_c + (-3.8, -2.0, 6.0)), 0.8,
                     P.mottled(P.hexrgb("#4a4650")), 0.1),
               S.rod(tuple(head_c + (2.4, -0.6, 2.4)), tuple(head_c + (3.8, -2.0, 6.0)), 0.8,
                     P.mottled(P.hexrgb("#4a4650")), 0.1)]
    for sx in (1, -1):
        barrel.append(S.hull([head_c + (sx * 1.6, 2.2, 0.8), head_c + (sx * 1.0, 2.2, 0.8),
                              head_c + (sx * 1.3, 2.6, -0.6)], S.mat(P.BONE)))
    weapon = Part("weapon", pivot=(0, -4.0, 9.5), boxes=barrel)
    wheels = [wheel("wheel_1", (5.6, -3.0, 5.2), 5.2, seed), wheel("wheel_2", (-5.6, -3.0, 5.2), 5.2, seed + 1)]
    root = Part("root").add(Part("body", boxes=body).add(weapon), *wheels, team_flag((-4.0, -15.0, 6.0), 10, seed))
    return _siege(key, "Gargoyle Bombard", "Bombard Cannon", root, scale=1.5)


# --------------------------------------------------------------------------- trebuchet

def gallows_trebuchet(key="trebuchet") -> Unit:
    """A trebuchet of black, rotten timber like a gallows: a coffin for the counterweight, a pumpkin in the sling,
    nooses hanging from the frame."""
    seed = 80
    wood, wood_up = WOOD(seed), WOOD_UP(seed)
    base = [S.block((5.0, -18.0, 2.0), (8.0, 18.0, 5.0), wood), S.block((-8.0, -18.0, 2.0), (-5.0, 18.0, 5.0), wood),
            S.block((-8.0, 14.0, 2.0), (8.0, 17.0, 5.0), WOOD(seed + 1)),
            S.block((-8.0, -17.0, 2.0), (8.0, -14.0, 5.0), WOOD(seed + 1))]
    frame = [S.block((5.0, -1.5, 5.0), (8.0, 1.5, 44.0), wood_up), S.block((-8.0, -1.5, 5.0), (-5.0, 1.5, 44.0), wood_up),
             S.block((-9.0, -1.0, 41.0), (9.0, 1.0, 43.0), wood)]
    for sx in (1, -1):
        x = sx * 6.5
        frame += [S.rod((x, 12.0, 5.0), (x, 1.5, 36.0), 0.9, wood_up), S.rod((x, -12.0, 5.0), (x, -1.5, 36.0), 0.9,
                                                                              wood_up)]
    rope = S.mat(P.hexrgb("#8a7a5a"))
    for x in (-8.6, 8.6):  # nooses
        frame += [S.rod((x, 1.4, 42.0), (x, 1.4, 34.0), 0.2, rope), S.torus((x, 1.4, 33.0), 1.0, 0.25, rope,
                                                                            rot=(0, 90, 0))]
    arm = [S.rod((0, -34.0, 42.0), (0, 10.0, 42.0), 1.4, P.bone(seed), 1.0)]  # a long thigh bone
    arm += [S.ball((0, -34.0, 42.0), 2.0, P.bone(seed)), S.ball((0, 10.0, 42.0), 2.2, P.bone(seed))]
    # the coffin counterweight
    cw = [S.rod((-2.0, 9.0, 42.0), (-2.0, 9.0, 34.0), 0.3, IRON(seed)),
          S.rod((2.0, 9.0, 42.0), (2.0, 9.0, 34.0), 0.3, IRON(seed)),
          S.hull([(x, y, z) for x, y in ((-3.5, 4.0), (3.5, 4.0), (5.0, 9.0), (3.8, 14.0), (-3.8, 14.0), (-5.0, 9.0))
                  for z in (22.0, 34.0)], WOOD(seed + 2)),
          S.block((-0.6, 6.0, 33.8), (0.6, 12.0, 34.6), IRON(seed)), S.block((-2.6, 10.0, 33.8), (2.6, 11.0, 34.6),
                                                                            IRON(seed))]
    sling_shapes = [S.rod((0, -33.0, 42.0), (0, -33.0, 35.0), 0.3, rope)] + PK.lantern((0, -33.0, 30.0), 2.6,
                                                                                       PK.SCARY, seed)
    arm_part = Part("arm", pivot=(0, 0, 42.0), boxes=arm).add(
        Part("counterweight", pivot=(0, 9.0, 42.0), boxes=cw), Part("sling", pivot=(0, -33.0, 42.0), boxes=sling_shapes))
    frame_part = Part("frame", pivot=(0, 0, 4.0), boxes=frame).add(arm_part, team_flag((6.5, 0.0, 44.0), 12, seed))
    wheels = [wheel(f"wheel_{k + 1}", at, 3.0, seed + k) for k, at in
              enumerate([(9.5, 14.0, 3.0), (-9.5, 14.0, 3.0), (9.5, -14.0, 3.0), (-9.5, -14.0, 3.0)])]
    root = Part("root").add(Part("base", boxes=base), frame_part, *wheels)
    return _siege(key, "Gallows Trebuchet", "Trebuchet", root, rig="trebuchet", scale=1.5)


# --------------------------------------------------------------------------- wagons and the train

def hearse_wagon(key="war_wagon") -> Unit:
    """A black hearse with glass sides showing a coffin, crossbow slits, a team-coloured plume, pulled by a
    nightmare."""
    seed = 90
    lacquer = P.metal(P.hexrgb("#1c1a20"), seed=seed, shine=0.75)
    glass = S.Material(color=lambda l: P.mix(P.hexrgb("#3a5a5a"), P.hexrgb("#a8e0d8"),
                                             0.4 * P.noise(l.p, 2.0, seed)), glow=0.25, shine=0.9, gloss=40)
    body = [S.block((-7.0, -12.0, 5.0), (7.0, 9.0, 7.0), lacquer),
            S.block((-6.4, -11.4, 7.0), (6.4, 8.4, 17.0), glass),
            S.block((-7.4, -12.4, 17.0), (7.4, 9.4, 19.0), lacquer)]
    for x in (-6.6, 6.6):
        for y in (-11.6, -1.4, 8.6):
            body.append(S.block((x - 0.5, y - 0.5, 7.0), (x + 0.5, y + 0.5, 17.0), lacquer))
    body.append(S.hull([(x, y, z) for x, y in ((-3.0, -10.0), (3.0, -10.0), (3.6, 1.0), (3.0, 6.0), (-3.0, 6.0),
                                               (-3.6, 1.0)) for z in (7.0, 10.5)], WOOD(seed)))  # the coffin inside
    for x in (-7.0, 7.0):  # crossbow slits
        body.append(S.block((x - 0.3, -6.0, 13.0), (x + 0.3, 2.0, 13.8), S.mat(P.hexrgb("#0a0608"))))
    gold = P.metal(P.GOLD, seed=seed, shine=0.8)
    for x in (-6.6, 6.6):
        for y in (-11.6, 8.6):
            body.append(S.ball((x, y, 20.0), 0.8, gold))
    body.append(S.Ellipsoid(C.team_cloth(0.55, seed), S._orient((0, -1.5, 21.5)), radii=(1.6, 1.6, 3.6)))
    weapon = Part("weapon", pivot=(0, 8.0, 14.0), boxes=[S.rod((0, 8.0, 13.6), (0, 11.0, 13.6), 0.5, WOOD(seed))])
    body.append(S.rod((0, 9.0, 6.0), (0, 22.0, 9.0), 0.5, WOOD(seed)))  # the shaft to the horse
    draught = CV.steed("nightmare", seed + 5)
    draught.name = "draught"
    draught.offset = (0, 30.0, 0)
    wheels = [wheel(f"wheel_{k + 1}", at, 4.5, seed + k) for k, at in
              enumerate([(8.0, 4.0, 4.5), (-8.0, 4.0, 4.5), (8.0, -8.0, 4.5), (-8.0, -8.0, 4.5)])]
    root = Part("root").add(Part("body", boxes=body).add(weapon), draught, *wheels)
    return _siege(key, "Hearse Wagon", "War Wagon", root, civ="Koreans", scale=1.0)


def ghost_train(key="trade_cart") -> Unit:
    """The ghost train: a black steam engine with a cowcatcher of bones, its firebox glowing green and green
    ghostly smoke from the chimney, pulling a coffin car with lanterns."""
    from .rails import TRAIN_SCALE
    seed = 95
    iron = P.metal(P.hexrgb("#24222a"), rust=0.3, seed=seed, shine=0.7)
    brass = P.metal(P.hexrgb("#a8823a"), seed=seed, shine=0.8)
    eng = [S.block((-5.5, 0.0, 3.0), (5.5, 16.0, 5.0), iron),  # the frame
           S.rod((0, 4.0, 10.0), (0, 15.0, 10.0), 4.6, iron),  # the boiler
           S.rod((0, 15.0, 10.0), (0, 15.6, 10.0), 4.2, brass),
           S.block((-5.0, 0.0, 5.0), (5.0, 5.0, 16.0), iron),  # the cab
           S.block((-5.6, -0.6, 16.0), (5.6, 5.6, 17.0), iron)]
    eng += [S.rod((0, 12.0, 14.0), (0, 12.0, 20.0), 1.4, iron, 2.2),  # the chimney, flaring
            S.torus((0, 12.0, 20.0), 2.0, 0.5, iron)]
    eng.append(S.ball((0, 15.8, 10.0), 1.2, S.mat(P.SLIME, glow=1.0)))  # the lamp
    for x in (-5.1, 5.1):  # glowing windows of the cab
        eng.append(S.block((x - 0.2, 1.4, 10.0), (x + 0.2, 3.6, 13.6), S.mat(P.SLIME, glow=0.85)))
    for k in range(5):  # the cowcatcher: ribs
        x = -4.0 + 2.0 * k
        eng.append(S.rod((x, 16.0, 4.0), (x * 0.5, 19.0, 1.0), 0.35, P.bone(seed)))
    body = Part("body", boxes=eng)
    for k in (1, 2):
        puff = S.Ellipsoid(C.ghostly(P.hexrgb("#9cf0b0"), seed + k), S._orient((0, 12.0, 23.0)),
                           radii=(2.6, 2.6, 2.2))
        puff.casts = False
        body.add(Part(f"smoke_{k}", boxes=[puff]))
    wood = WOOD(seed)
    car = [S.block((-5.5, -22.0, 3.0), (5.5, -4.0, 5.0), iron),
           S.hull([(x, y, z) for x, y in ((-3.6, -20.0), (3.6, -20.0), (4.6, -11.0), (3.6, -6.0), (-3.6, -6.0),
                                          (-4.6, -11.0)) for z in (5.0, 11.0)], wood),
           S.hull([(x, y, z) for x, y in ((-3.9, -20.3), (3.9, -20.3), (4.9, -11.0), (3.9, -5.7), (-3.9, -5.7),
                                          (-4.9, -11.0)) for z in (11.0, 12.2)], WOOD(seed + 1)),
           S.block((-0.5, -17.0, 12.2), (0.5, -8.0, 13.0), brass), S.block((-2.6, -10.5, 12.2), (2.6, -9.5, 13.0),
                                                                            brass),
           S.rod((0, -4.0, 4.0), (0, 0.0, 4.0), 0.5, iron)]
    for y in (-21.0, -5.0):
        car += [S.rod((4.6, y, 5.0), (4.6, y, 12.0), 0.25, iron),
                S.block((4.0, y - 0.6, 12.0), (5.2, y + 0.6, 13.4), S.mat(P.SLIME, glow=0.9))]
    wagon = Part("wagon", boxes=car).add(team_flag((-4.6, -21.0, 5.0), 12, seed))
    body.add(wagon)
    wheels = []
    k = 1
    for y in (13.0, 5.0, -7.0, -19.0):
        for x in (6.0, -6.0):
            wheels.append(wheel(f"wheel_{k}", (x, y, 2.6), 2.6, seed + k, spokes=6))
            k += 1
    root = Part("root").add(body, *wheels)
    return _siege(key, "Ghost Train", "Trade Cart", root, attack="none", scale=TRAIN_SCALE)


SIEGE = {
    "battering_ram": lambda: coffin_ram("battering_ram", "Coffin Ram", "Battering Ram", 0),
    "capped_ram": lambda: coffin_ram("capped_ram", "Iron Coffin Ram", "Capped Ram", 1),
    "siege_ram": lambda: coffin_ram("siege_ram", "Siege Coffin Ram", "Siege Ram", 2),
    "mangonel": lambda: bone_catapult("mangonel", None, "Mangonel", 0),
    "onager": lambda: bone_catapult("onager", None, "Onager", 1),
    "siege_onager": lambda: bone_catapult("siege_onager", None, "Siege Onager", 2),
    "scorpion": lambda: bone_ballista("scorpion", "Bone Ballista", "Scorpion"),
    "heavy_scorpion": lambda: bone_ballista("heavy_scorpion", "Heavy Bone Ballista", "Heavy Scorpion", True),
    "bombard_cannon": gargoyle_bombard,
    "trebuchet": gallows_trebuchet,
    "war_wagon": hearse_wagon,
    "trade_cart": ghost_train,
}
