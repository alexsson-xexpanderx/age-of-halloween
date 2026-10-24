"""Animals and the easter eggs, on the old rigs' parts:

    quadruped      "torso", "head", "tail", "leg_fr/fl/br/bl"
    cavalry        the horse: "horse_body", "horse_head", "tail", legs
    flyer          "torso", "wing_r/l", "tip_r/l"
    swimmer        "torso", "tail"
    biped_animal   "torso", "head", "leg_r/l"
    ghost          "body" and its skirt's "foot*" parts
    wheeled        "body", "wheel_1..4" (the Cobra car)
    dragon         "torso", "neck", "head", "jaw", "fire_long", "fire_short", "tail1..5", "wing_r/l", "tip_r/l", "legs"

The dragon keeps the old one's height in the air and its mouth's place (easter.DRAGON_HEIGHT, DRAGON_MOUTH): the
game's data is set for them.
"""
from __future__ import annotations

import math

import numpy as np

from . import cavalry as CV
from . import creatures as C
from . import paint as P
from . import sculpt as S
from .easter import DRAGON_HEIGHT, DRAGON_MOUTH, DRAGON_SCALE
from .engines import IRON, WOOD, wheel
from .geometry import Part
from .units import Unit

RED_EYE = P.hexrgb("#ff3a1a")
GREEN_EYE = RED_EYE  # the player asked for red eyes instead of green ones


def _eyes(skin, glow, x=1.1, z=0.5, r=0.42):
    return C.eyes_material(skin, [(-x, z, r, r * 0.85), (x, z, r, r * 0.85)], glow, 0.6)


def quadruped(key, name, replaces, skin, size=1.0, eye=RED_EYE, attack="none", tail=True, horns=None, tusks=False,
              spikes=False, mane=None, bony=False, legs=1.0, neck=0.0, snout=1.0, ears=True, wool=None, scale=1.0):
    """A four-legged animal facing +y; `size` about 1 for a wolf (back at 11)."""
    k = size
    back = 11.0 * k * legs
    body_r = (3.4 * k, 7.5 * k, 3.6 * k)
    torso_shapes = []
    if bony:
        from .undead import _ribs_material
        torso_shapes += [S.ball((0, 1.0 * k, back - 1.0 * k), (body_r[0], body_r[1] * 0.7, body_r[2]),
                                _ribs_material(7)).clip((0, 0, -1), -(back - 3.0 * k) + back - 1.0 * k),
                         S.ball((0, -5.0 * k, back - 0.6 * k), (2.6 * k, 2.2 * k, 2.0 * k), skin)]
        torso_shapes += [S.ball((0, y, back + 2.4 * k), 0.6 * k, skin) for y in np.linspace(-6 * k, 6 * k, 7)]
    else:
        torso_shapes.append(S.Ellipsoid(wool if wool is not None else skin, S._orient((0, 0, back - 0.5 * k)),
                                        radii=body_r))
    if spikes:
        for y in np.linspace(-5 * k, 5 * k, 6):
            torso_shapes.append(S.rod((0, y, back + 2.8 * k), (0, y - 1.0 * k, back + 5.0 * k), 0.7 * k,
                                      S.mat(P.hexrgb("#1a1214")), 0.05))
    torso = Part("torso", pivot=(0, 0, back), boxes=torso_shapes)
    # head
    hc = np.array([0, 8.6 * k, back + 2.0 * k + neck * k])
    head_shapes = S.limb((0, 5.5 * k, back), tuple(hc - (0, 1.2 * k, 0.6 * k)), 1.6 * k, skin, 1.4 * k) if not bony \
        else [S.ball(tuple(np.array([0, 5.5 * k, back]) + (hc - np.array([0, 5.5 * k, back])) * f), 0.7 * k, skin)
              for f in np.linspace(0, 1, 4)]
    head_m = _eyes(skin, eye, 1.0 * k, 0.5 * k, 0.4 * k)
    head_shapes.append(S.Ellipsoid(head_m, S._orient(hc), radii=(2.2 * k, 2.4 * k, 2.1 * k)))
    if snout:
        head_shapes.append(S.rod(tuple(hc + (0, 1.2 * k, -0.6 * k)), tuple(hc + (0, (2.0 + 2.4 * snout) * k,
                                                                                  -1.0 * k)), 1.2 * k, skin, 0.8 * k))
        head_shapes.append(S.ball(tuple(hc + (0, (2.1 + 2.4 * snout) * k, -0.8 * k)), 0.45 * k,
                                  S.mat(P.hexrgb("#141012"))))
    if ears:
        for sx in (-1, 1):
            head_shapes.append(S.rod(tuple(hc + (sx * 1.2 * k, -0.4 * k, 1.4 * k)),
                                     tuple(hc + (sx * 1.7 * k, -0.8 * k, 3.4 * k)), 0.55 * k, skin, 0.05))
    if horns == "antlers":
        bone = P.bone(9)
        for sx in (-1, 1):
            base = hc + (sx * 1.2 * k, -0.2 * k, 1.6 * k)
            a = base + (sx * 2.6 * k, -1.0 * k, 4.2 * k)
            b = a + (sx * 1.6 * k, -1.2 * k, 3.4 * k)
            head_shapes += [S.rod(tuple(base), tuple(a), 0.45 * k, bone, 0.35 * k),
                            S.rod(tuple(a), tuple(b), 0.35 * k, bone, 0.1 * k)]
            for f, d in ((0.5, (sx * 0.4, 1.4, 2.0)), (1.0, (-sx * 0.6, 1.0, 2.6))):
                p = base + (a - base) * f
                head_shapes.append(S.rod(tuple(p), tuple(p + np.array(d) * k), 0.25 * k, bone, 0.05))
    elif horns == "ram":
        horn = P.mottled(P.hexrgb("#4a4038"), P.hexrgb("#1e1814"), 1.0, 0.4)
        for sx in (-1, 1):
            head_shapes.append(S.torus(tuple(hc + (sx * 2.0 * k, -0.4 * k, 0.6 * k)), 1.4 * k, 0.55 * k, horn,
                                       rot=(0, 90, 0)))
    if tusks:
        for sx in (-1, 1):
            head_shapes.append(S.rod(tuple(hc + (sx * 0.8 * k, 3.4 * k, -1.6 * k)),
                                     tuple(hc + (sx * 1.2 * k, 4.6 * k, 0.2 * k)), 0.35 * k,
                                     S.mat(P.hexrgb("#e8dcc0")), 0.05))
    if mane is not None:
        head_shapes.append(S.ball(tuple(hc + (0, -1.6 * k, 0.4 * k)), (2.6 * k, 2.4 * k, 2.6 * k), mane))
    head = Part("head", pivot=(0, 5.5 * k, back), boxes=head_shapes)
    torso.add(head)
    if tail:
        t0 = np.array([0, -7.2 * k, back])
        tail_shapes = S.limb(tuple(t0), tuple(t0 + (0, -3.0 * k, -3.6 * k)), 0.8 * k, skin, 0.3 * k) if not bony else \
            [S.ball(tuple(t0 + (0, -f * 3.0 * k, -f * 3.6 * k)), 0.5 * k, skin) for f in np.linspace(0, 1, 5)]
        torso.add(Part("tail", pivot=tuple(t0), boxes=tail_shapes))

    def leg(name, x, y):
        hip = np.array([x, y, back - 1.0 * k])
        knee = hip + (0, (0.6 if y > 0 else -0.6) * k, -(back - 1.0 * k) * 0.5)
        foot = np.array([x, y, 0.6 * k])
        if bony:
            b = C.body("skeleton", 7, skin=skin)
            shapes = C._limb_shapes(b, tuple(hip), tuple(knee), 0.7 * k, 0.55 * k) + \
                C._limb_shapes(b, tuple(knee), tuple(foot), 0.55 * k, 0.45 * k)
        else:
            shapes = S.limb(tuple(hip), tuple(knee), 1.1 * k, skin, 0.85 * k) + \
                S.limb(tuple(knee), tuple(foot), 0.85 * k, skin, 0.6 * k)
        return Part(name, pivot=tuple(hip), boxes=shapes)
    lx = 2.0 * k
    root = Part("root").add(torso, leg("leg_fr", lx, 5.0 * k), leg("leg_fl", -lx, 5.0 * k),
                            leg("leg_br", lx, -5.0 * k), leg("leg_bl", -lx, -5.0 * k))
    return Unit(key, name, replaces, "quadruped", root, group="animal", attack=attack, scale=scale)


def flyer(key, name, replaces, kind="bat", scale=1.0) -> Unit:
    """A bat (leathery wings with finger bones) or a raven (black feathers), flying high, facing +y."""
    h = 34.0
    if kind == "bat":
        skin = C.fur(P.hexrgb("#4a3a3a"), P.hexrgb("#1a1214"), 3)
        wing_m = S.Material(color=lambda l: P.mix(P.hexrgb("#5a2e2e"), P.hexrgb("#1e0e10"),
                                                  0.5 + 0.4 * P.noise(l.p, 1.5, 4)), vivid=0.3)
    else:
        skin = S.Material(color=lambda l: P.mix(P.hexrgb("#24222e"), P.hexrgb("#0a0a10"),
                                                0.5 * P.noise(l.p * np.array([1, 3, 1]), 1.0, 5)), shine=0.6,
                          gloss=30)
        wing_m = skin
    head_m = _eyes(skin, RED_EYE if kind == "bat" else P.hexrgb("#ffd040"), 0.7, 0.3, 0.3)
    torso_shapes = [S.Ellipsoid(skin, S._orient((0, 0, h)), radii=(1.8, 3.4, 1.8)),
                    S.Ellipsoid(head_m, S._orient((0, 3.6, h + 0.6)), radii=(1.4, 1.5, 1.4))]
    if kind == "bat":
        for sx in (-1, 1):
            torso_shapes.append(S.rod((sx * 0.7, 3.6, h + 1.6), (sx * 1.3, 3.2, h + 3.4), 0.45, skin, 0.05))
    else:
        torso_shapes += [S.rod((0, 4.6, h + 0.4), (0, 6.6, h - 0.2), 0.5, S.mat(P.hexrgb("#1a1a1e")), 0.08),
                         S.hull([(-1.4, -3.0, h), (1.4, -3.0, h), (0, -6.6, h - 0.4), (0, -6.0, h + 0.4)], skin)]
    torso = Part("torso", pivot=(0, 0, h), boxes=torso_shapes)
    for side in (1, -1):
        x0, x1 = 1.4 * side, 7.0 * side
        if kind == "bat":
            inner = [S.hull([(x0, 1.0, h), (x1, 1.4, h + 0.4), (x1, -3.0, h), (x0, -2.6, h), (x0, -0.8, h + 0.2)],
                            wing_m), S.rod((x0, 1.0, h + 0.2), (x1, 1.4, h + 0.4), 0.25, skin)]
            outer = [S.hull([(x1, 1.4, h + 0.4), (x1 + 6.0 * side, 0.6, h + 0.8), (x1 + 4.0 * side, -2.8, h),
                             (x1, -3.0, h)], wing_m),
                     S.rod((x1, 1.4, h + 0.4), (x1 + 6.0 * side, 0.6, h + 0.8), 0.2, skin)]
        else:
            inner = [S.hull([(x0, 1.6, h), (x1, 1.0, h + 0.3), (x1, -3.4, h), (x0, -2.4, h), (x0, 0, h + 0.4)],
                            wing_m)]
            outer = [S.hull([(x1, 1.0, h + 0.3), (x1 + 5.6 * side, -0.4, h + 0.5), (x1 + 5.0 * side, -3.4, h),
                             (x1, -3.4, h)], wing_m)]
        wing = Part("wing_r" if side > 0 else "wing_l", pivot=(x0, 0, h), boxes=inner)
        wing.add(Part("tip_r" if side > 0 else "tip_l", pivot=(x1, 0, h), boxes=outer))
        torso.add(wing)
    root = Part("root").add(torso)
    return Unit(key, name, replaces, "flyer", root, group="animal", attack="none", scale=scale)


def bone_fish(key, name, replaces, length=8.0, scale=1.6, ghost=False) -> Unit:
    """A fish skeleton, or a glowing ghost fish: spine, ribs and a skull, swimming."""
    m = C.ghostly(P.hexrgb("#9cf0d8"), 3) if ghost else P.bone(3)
    L = length
    torso = [S.Ellipsoid(_eyes(m, GREEN_EYE, 0.6, 0.3, 0.3), S._orient((0, L * 0.45, 0.0)),
                         radii=(L * 0.13, L * 0.2, L * 0.17)),
             S.rod((0, -L * 0.4, 0), (0, L * 0.3, 0), L * 0.035, m)]
    for k in range(6):
        y = L * (0.25 - 0.1 * k)
        r = L * 0.13 * (1 - 0.1 * k)
        torso.append(S.torus((0, y, 0), r, L * 0.018, m, rot=(90, 0, 0)))
    tail = [S.hull([(0, -L * 0.4, 0), (0, -L * 0.62, L * 0.18), (0, -L * 0.62, -L * 0.18), (0.08, -L * 0.45, 0),
                    (-0.08, -L * 0.45, 0)], m)]
    root = Part("root").add(Part("torso", boxes=torso).add(Part("tail", pivot=(0, -L * 0.4, 0), boxes=tail)))
    return Unit(key, name, replaces, "swimmer", root, group="animal", attack="none", scale=scale)


def vulture(key="turkey") -> Unit:
    """The turkey: a bald black vulture with a red, wrinkled head, its eyes glowing."""
    feathers = S.Material(color=lambda l: P.mix(P.hexrgb("#2a2630"), P.hexrgb("#0e0c12"),
                                                0.5 * P.noise(l.p * np.array([1, 1, 3]), 1.0, 2)), shine=0.4)
    neck_m = C.flesh(P.hexrgb("#a83a30"), P.hexrgb("#4a1410"), 2)
    torso = Part("torso", pivot=(0, 0, 8.0), boxes=[
        S.Ellipsoid(feathers, S._orient((0, 0, 9.0)), radii=(3.0, 4.2, 3.4)),
        S.hull([(-2.8, -2.0, 10.0), (2.8, -2.0, 10.0), (-3.6, -7.0, 13.0), (3.6, -7.0, 13.0), (0, -5.0, 9.0)],
               feathers),
        S.torus((0, 2.6, 11.8), 1.4, 0.7, S.mat(P.hexrgb("#ece4d8")))])  # a ruff of white down
    head = Part("head", pivot=(0, 3.0, 12.0), boxes=[
        *S.limb((0, 3.0, 12.0), (0, 4.4, 15.4), 0.7, neck_m, 0.6),
        S.Ellipsoid(_eyes(neck_m, P.hexrgb("#ffd040"), 0.55, 0.3, 0.25), S._orient((0, 4.8, 16.2)),
                    radii=(1.0, 1.2, 1.0)),
        S.rod((0, 5.6, 16.0), (0, 7.2, 15.2), 0.45, S.mat(P.hexrgb("#c8b896")), 0.1)])
    torso.add(head)

    def leg(name, x):
        return Part(name, pivot=(x, 0, 7.0), boxes=S.limb((x, 0, 7.0), (x, 0.4, 0.6), 0.35,
                                                          S.mat(P.hexrgb("#6a5a4a"))) +
                    [S.rod((x, 0.4, 0.4), (x + dx, 1.8, 0.2), 0.2, S.mat(P.hexrgb("#6a5a4a"))) for dx in (-0.6, 0.6)])
    root = Part("root").add(torso, leg("leg_r", 1.2), leg("leg_l", -1.2))
    return Unit(key, "Vulture", "Turkey", "biped_animal", root, group="animal", attack="none")


def nightmare_horse(key="horse") -> Unit:
    """A riderless nightmare (gaia's horses have no saddle)."""
    root = CV.steed("nightmare", 91, saddle=False)
    return Unit(key, "Nightmare", "Horse", "cavalry", root, group="animal", attack="none")


ANIMALS = {
    "jaguar": lambda: quadruped("jaguar", "Shadow Panther", "Jaguar",
                                C.fur(P.hexrgb("#2a2830"), P.hexrgb("#0e0c12"), 4), 1.1, P.hexrgb("#9cff6a"),
                                attack="bite", snout=0.6),
    "horse": nightmare_horse,
    "hawk": lambda: flyer("hawk", "Giant Bat", "Hawk", "bat"),
    "macaw": lambda: flyer("macaw", "Raven", "Macaw", "raven", 1.3),
    "marlin": lambda: bone_fish("marlin", "Ghost Marlin", "Marlin", 16.0, 1.3, ghost=True),
    "fish_perch": lambda: bone_fish("fish_perch", "Bone Perch", "Fish (Perch)"),
    "fish_salmon": lambda: bone_fish("fish_salmon", "Bone Salmon", "Fish (Salmon)", 9.0),
    "fish_tuna": lambda: bone_fish("fish_tuna", "Ghost Tuna", "Fish (Tuna)", 10.0, ghost=True),
    "fish_dorado": lambda: bone_fish("fish_dorado", "Bone Dorado", "Fish (Dorado)", 9.0),
    "fish_snapper": lambda: bone_fish("fish_snapper", "Ghost Snapper", "Fish (Snapper)", 8.0, ghost=True),
    "fish_shore": lambda: bone_fish("fish_shore", "Bone Fish", "Shore Fish", 7.0),
    "sheep": lambda: quadruped("sheep", "Black Sheep", "Sheep", C.flesh(P.hexrgb("#3a3036"), P.hexrgb("#1a1418"), 5),
                               0.85, RED_EYE, snout=0.5, horns="ram", legs=0.9,
                               wool=S.Material(color=lambda l: P.mix(P.hexrgb("#2a2630"), P.hexrgb("#0e0c10"),
                                                                     P.noise(l.p, 0.8, 6)), bump=None)),
    "turkey": vulture,
    "deer": lambda: quadruped("deer", "Skeletal Stag", "Deer", P.bone(8), 1.15, GREEN_EYE, horns="antlers", bony=True,
                              legs=1.25, neck=1.4, snout=0.9),
    "wild_boar": lambda: quadruped("wild_boar", "Demon Boar", "Wild Boar",
                                   C.fur(P.hexrgb("#4a2a24"), P.hexrgb("#1a0c0a"), 9), 1.15, RED_EYE, attack="bite",
                                   tusks=True, spikes=True, legs=0.75, snout=0.9),
    "javelina": lambda: quadruped("javelina", "Imp Boar", "Javelina",
                                  C.fur(P.hexrgb("#5a3428"), P.hexrgb("#24100c"), 10), 0.85, RED_EYE, attack="bite",
                                  tusks=True, spikes=True, legs=0.75, snout=0.8),
    "wolf": lambda: quadruped("wolf", "Dire Wolf", "Wolf", C.fur(P.hexrgb("#4a4650"), P.hexrgb("#16141a"), 11), 1.0,
                              P.hexrgb("#ffd040"), attack="bite", snout=1.0,
                              mane=C.fur(P.hexrgb("#3a3640"), P.hexrgb("#121016"), 12)),
}


# --------------------------------------------------------------------------- easter eggs

def sheet_ghost(key="ghost") -> Unit:
    """The VDML cheat's unit: a bedsheet ghost in the owner's colour, black eye holes, the hem rippling."""
    def holes(look):
        p = look.p
        eyes = (((p[:, 0] - 1.6) / 0.9) ** 2 + ((p[:, 2] - 12.6) / 1.3) ** 2 < 1) | \
               (((p[:, 0] + 1.6) / 0.9) ** 2 + ((p[:, 2] - 12.6) / 1.3) ** 2 < 1)
        mouth = ((p[:, 0] / 1.3) ** 2 + ((p[:, 2] - 9.6) / 0.9) ** 2 < 1)
        dark = (eyes | mouth) & (p[:, 1] > 2.0)
        return np.where(dark, 0.02, 0.82 + 0.12 * np.sin(np.arctan2(p[:, 1], p[:, 0]) * 6))
    sheet = P.team(1.0, holes)
    body = Part("body", boxes=[S.ball((0, 0, 11.5), (5.2, 5.0, 4.8), sheet).clip((0, 0, -1), -10.5),
                               S.rod((0, 0, 11.5), (0, 0, 4.0), 5.2, sheet, 6.4, capped=False)])
    skirt = []
    for k in range(8):
        a = 2 * math.pi * k / 8
        c = (math.cos(a) * 5.6, math.sin(a) * 5.6, 3.4)
        skirt.append(Part(f"foot{k % 2}_{k}", boxes=[S.ball(c, (2.0, 2.0, 1.6), sheet)]))
    root = Part("root", offset=(0, 0, 2)).add(body, *skirt)
    return Unit(key, "Ghost", "VDML (cheat)", "ghost", root, group="easter", attack="none", scale=1.2)


def hotrod_hearse(key="cobra_car") -> Unit:
    """The Cobra Car cheat: a hot-rod hearse in the owner's colour with flames painted on its sides, a coffin in the
    back, chrome pipes and a big engine poking out of the hood."""
    def flames(look):
        p = look.p
        f = (p[:, 2] < 6.0 + 2.2 * np.sin(p[:, 1] * 0.8) + 1.4 * np.sin(p[:, 1] * 2.1)) & (p[:, 1] > -6)
        return np.where(f, 1.4, 0.55)
    paint = P.team(0.55, flames)
    paint.shine, paint.gloss = 0.8, 40
    chrome = P.metal(P.hexrgb("#b08a3a"), seed=3, shine=1.0)  # brass, not chrome
    glass = S.Material(color=lambda l: P.mix(P.hexrgb("#1e2a30"), P.hexrgb("#5a7a80"), 0.4 * P.noise(l.p, 2.0, 4)),
                       shine=0.9, gloss=50)
    body = [S.hull([(x, y, z) for x in (-7.5, 7.5) for y, z0, z1 in ((-19.0, 4.0, 10.0), (18.0, 4.0, 8.5))
                    for z in (z0, z1)], paint),
            S.hull([(x, y, z) for x in (-6.5, 6.5) for y in (-18.0, -2.0) for z in (10.0, 16.0)] +
                   [(x, 0.5, 10.0) for x in (-6.5, 6.5)], paint)]
    for x in (-6.6, 6.6):  # side windows with lace curtains
        body.append(S.block((x - 0.15, -16.5, 11.0), (x + 0.15, -3.5, 15.0), glass))
    body.append(S.hull([(x, y, z) for x, y in ((-3.0, -16.0), (3.0, -16.0), (3.6, -11.0), (3.0, -5.0), (-3.0, -5.0),
                                               (-3.6, -11.0)) for z in (10.0, 12.0)], WOOD(5)))  # the coffin
    body += [S.block((-3.0, 10.0, 8.5), (3.0, 16.0, 11.5), chrome),  # the engine, blower on top
             S.block((-1.8, 11.0, 11.5), (1.8, 15.0, 13.5), chrome)]
    for x in (-8.0, 8.0):
        body.append(S.rod((x, 12.0, 4.5), (x, -14.0, 3.6), 0.6, chrome))
    body.append(S.Ellipsoid(S.mat(P.hexrgb("#ffe080"), glow=1.0), S._orient((5.0, 18.2, 7.0)), radii=(1.1, 0.4, 1.0)))
    body.append(S.Ellipsoid(S.mat(P.hexrgb("#ffe080"), glow=1.0), S._orient((-5.0, 18.2, 7.0)), radii=(1.1, 0.4, 1.0)))
    windshield = Part("windshield", boxes=[S.block((-6.5, 0.2, 10.0), (6.5, 0.8, 14.5), glass, rot=(-25, 0, 0))])
    wheels = [wheel(f"wheel_{k + 1}", at, r, 7 + k) for k, (at, r) in
              enumerate((((8.2, 12.0, 3.6), 3.6), ((-8.2, 12.0, 3.6), 3.6), ((8.4, -13.0, 4.4), 4.4),
                         ((-8.4, -13.0, 4.4), 4.4)))]
    root = Part("root").add(Part("body", boxes=body), windshield, *wheels)
    return Unit(key, "Hot-Rod Hearse", "Cobra Car (cheat)", "wheeled", root, group="easter", attack="none", scale=1.5)


def bone_dragon(key="dragon") -> Unit:
    """A dragon's skeleton, flying on tattered wings of the owner's colour, green fire in its ribs and eyes, breathing
    green fire. The old dragon's sizes: body centre DRAGON_HEIGHT up, mouth DRAGON_MOUTH from it."""
    bone = P.bone(13)
    from .undead import _ribs_material
    my, mz = DRAGON_MOUTH
    torso_shapes = [S.ball((0, 0, 0), (6.0, 13.0, 5.5), _ribs_material(13)),
                    S.Ellipsoid(S.mat(P.SLIME, glow=0.9), S._orient((0, 2.0, -0.5)), radii=(2.6, 5.0, 2.4))]
    torso_shapes += [S.ball((0, y, 5.6 - abs(y) * 0.05), (1.3, 1.2, 1.4), bone) for y in np.linspace(-12, 12, 9)]
    for y in np.linspace(-10, 10, 5):  # spikes down the back
        torso_shapes.append(S.rod((0, y, 6.4), (0, y - 1.5, 9.4), 0.7, bone, 0.05))
    torso = Part("torso", boxes=torso_shapes)
    # the neck: vertebrae towards the head, which sits so its mouth is at (my, mz)
    head_c = np.array([0.0, my - 9.0, mz + 1.5])
    neck_pts = [np.array([0, 13.0, 1.0]) + (head_c - np.array([0, 15.0, 1.0])) * f for f in np.linspace(0, 1, 7)]
    neck = Part("neck", pivot=(0, 14.0, 0), boxes=[S.ball(tuple(p), 1.6 - 0.5 * f, bone)
                                                     for p, f in zip(neck_pts, np.linspace(0, 1, 7))])
    skull_m = C.eyes_material(bone, [(-2.6, 1.0, 1.0, 0.8), (2.6, 1.0, 1.0, 0.8)], RED_EYE, 0.6)
    head = Part("head", pivot=tuple(head_c - (0, 2.0, 0)), boxes=[
        S.Ellipsoid(skull_m, S._orient(head_c), radii=(4.4, 5.4, 3.8)),
        S.hull([head_c + (sx * 2.6, 3.0, -0.8) for sx in (-1, 1)] + [head_c + (sx * 1.6, 9.2, -1.0) for sx in (-1, 1)] +
               [head_c + (sx * 2.2, 3.0, 1.6) for sx in (-1, 1)] + [head_c + (sx * 1.2, 9.0, 0.6) for sx in (-1, 1)],
               bone)])
    for sx in (-1, 1):  # horns sweeping back
        head.boxes += [S.rod(tuple(head_c + (sx * 2.4, -2.0, 2.6)), tuple(head_c + (sx * 4.6, -9.0, 5.0)), 1.1, bone,
                             0.08)]
    for k in range(5):  # upper teeth
        x = -1.6 + 0.8 * k
        head.boxes.append(S.rod(tuple(head_c + (x * 1.1, 6.0 + k % 2, -0.9)), tuple(head_c + (x, 6.2 + k % 2, -2.2)),
                                0.3, S.mat(P.BONE), 0.03))
    jaw = Part("jaw", pivot=tuple(head_c + (0, 2.0, -1.4)), boxes=[
        S.hull([head_c + (sx * 2.4, 2.0, -1.4) for sx in (-1, 1)] + [head_c + (sx * 1.4, 9.0, -1.6) for sx in (-1, 1)] +
               [head_c + (sx * 2.0, 2.0, -3.0) for sx in (-1, 1)] + [head_c + (sx * 1.0, 8.6, -2.6) for sx in (-1, 1)],
               bone)])
    head.add(jaw)
    mouth = np.array([0.0, my, mz])

    def fire(length):
        out = []
        for k in range(5):
            f = k / 4
            c = mouth + (0, 4.0 + f * length, -f * 2.0)
            r = 1.6 + f * 3.0
            col = P.mix(P.hexrgb("#e8ffd8"), P.hexrgb("#3ad04a"), f)
            e = S.Ellipsoid(S.mat(col, glow=1.0), S._orient(c), radii=(r, 3.2 + f * 2.0, r))
            e.casts = False
            e.ink = False
            out.append(e)
        return out
    head.add(Part("fire_long", boxes=fire(24.0)), Part("fire_short", boxes=fire(6.0)[:2]))
    neck.add(head)
    # the tail: five links of vertebrae
    tail, y = torso, -13.0
    for k in range(5):
        link = Part(f"tail{k + 1}", pivot=(0, y, 0), boxes=[S.ball((0, y - d, 0), 1.3 - 0.2 * k, bone)
                                                           for d in (1.5, 3.5, 5.5)] +
                    ([S.hull([(0, y - 6.0, 0), (-3.0, y - 9.0, 0), (3.0, y - 9.0, 0), (0, y - 11.0, 0.4),
                              (0, y - 8.0, 0.6)], bone)] if k == 4 else []))
        tail.add(link)
        tail, y = link, y - 7.0

    def membrane(seed):
        def torn(look):
            p = look.p
            hole = P.noise(p, 2.2, seed) > 0.68
            return np.where(hole, 0.08, 0.45 + 0.15 * P.noise(p, 4.0, seed + 1))
        return P.team(1.0, torn)
    for side in (1, -1):
        x0, x1 = 6.0 * side, 26.0 * side
        arm_bone = [S.rod((x0, 6.0, 3.0), (x1, 5.0, 3.0), 1.0, bone, 0.7)]
        arm_web = [S.hull([(x0, 5.0, 3.0), (x1, 4.6, 3.0), (x1, -10.0, 2.6), (x0, -12.0, 2.6), (x0 + 6 * side, -6.0, 2.4)],
                          membrane(20 + side))]
        for f in (0.35, 0.7):  # finger bones through the membrane
            x = x0 + (x1 - x0) * f
            arm_bone.append(S.rod((x, 5.4, 3.0), (x + 2.0 * side, -11.0, 2.8), 0.35, bone, 0.15))
        wing = Part("wing_r" if side > 0 else "wing_l", pivot=(x0, 6.0, 3.0), boxes=arm_bone + arm_web)
        x2 = x1 + 18.0 * side
        tip = Part("tip_r" if side > 0 else "tip_l", pivot=(x1, 5.0, 3.0), boxes=[
            S.rod((x1, 5.0, 3.0), (x2, 2.0, 3.6), 0.7, bone, 0.3),
            S.hull([(x1, 4.6, 3.0), (x2, 2.0, 3.6), (x2 - 4.0 * side, -6.0, 3.0), (x1, -10.0, 2.6)],
                   membrane(30 + side)),
            S.rod((x1 + 9.0 * side, 3.8, 3.2), (x1 + 8.0 * side, -9.0, 2.8), 0.3, bone, 0.1)])
        wing.add(tip)
        torso.add(wing)
    claw = S.mat(P.hexrgb("#2a2420"))
    legs = []
    for x, y in ((3.5, 8.0), (-3.5, 8.0), (4.0, -8.0), (-4.0, -8.0)):
        legs += S.limb((x, y, -3.0), (x, y + 1.0, -8.0), 0.9, bone, 0.6) + \
            [S.rod((x + dx, y + 1.0, -8.0), (x + dx * 1.6, y + 2.8, -9.4), 0.25, claw, 0.03) for dx in (-0.6, 0, 0.6)]
    torso.add(Part("legs", boxes=legs), neck)
    root = Part("root", offset=(0, 0, DRAGON_HEIGHT)).add(torso)
    return Unit(key, "Bone Dragon", "Adv. Heavy Crossbowman", "dragon", root, group="easter", attack="fire",
                scale=DRAGON_SCALE)


def green_fireball(key="fireball") -> Unit:
    """The dragon's shot: a ball of green fire with a burning skull in it, licks of flame flickering round it."""
    from .graveyard import skull
    core = S.ball((0, 0, 0), 3.6, S.mat(P.hexrgb("#7cff6a"), glow=1.0))
    core.ink = False
    halo = S.ball((0, -1.0, 0), (4.4, 5.4, 4.4), S.mat(P.hexrgb("#2aa040"), glow=0.9))
    halo.ink = False
    sk = skull((0, 1.6, -1.2), 1.4, 3)
    licks_a = [S.rod((0, -3.0, 0), (1.5, -8.0, 2.0), 1.2, S.mat(P.hexrgb("#b8ff9a"), glow=1.0), 0.05),
               S.rod((0, -3.0, 0), (-2.0, -7.0, -1.6), 1.0, S.mat(P.hexrgb("#b8ff9a"), glow=1.0), 0.05)]
    licks_b = [S.rod((0, -3.0, 0), (-1.2, -8.4, 2.4), 1.2, S.mat(P.hexrgb("#b8ff9a"), glow=1.0), 0.05),
               S.rod((0, -3.0, 0), (2.2, -6.8, -1.2), 1.0, S.mat(P.hexrgb("#b8ff9a"), glow=1.0), 0.05)]
    root = Part("root").add(Part("ball", boxes=[halo, core] + sk), Part("flame_a", boxes=licks_a),
                            Part("flame_b", boxes=licks_b))
    return Unit(key, "Green fire", "the dragon's projectile", "fireball", root, group="easter", attack="none")


EASTER = {"ghost": sheet_ghost, "cobra_car": hotrod_hearse, "dragon": bone_dragon}
