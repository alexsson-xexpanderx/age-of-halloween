"""The creature kit: every walking person of Age of Halloween is a body, gear and what they hold.

Bodies stand on the joints Age of Minecraft's people had, so animation.py's rigs move them unchanged: "root" with
"torso" (pivot at the hips) carrying "head", "arm_r", "arm_l" (pivots at the shoulders), and "leg_r", "leg_l" (pivots
at the hips). What a hand holds is a part "item_r" / "item_l" pivoting at the hand; a two-handed crossbow or gun is
"item_c" on the torso; a load carried in both arms is "carry" (shown only while carrying); a cape is "cape".

A body kind sets the skin, head, limbs and proportions:

    skeleton   bone, ribs, a skull with glowing eyes          zombie    rotting grey-green flesh, torn clothes
    ghoul      gaunt, blue-grey, long arms, hunched           imp       small, red-black, horns, a tail
    werewolf   fur, a wolf's head, claws                      mummy     bandages wound round, glowing eyes
    vampire    pale, slicked hair, red eyes                   batman    a bat's head and folded wings (Camazotz)
    calavera   a painted sugar skull (Day of the Dead)        ghost     pale and glowing, floating, no legs
    jiangshi   a hopping corpse in a robe and a hat           knight    a death knight: black plate, red visor
    draugr     a Norse corpse, icy blue eyes                  banshee   a floating wailing woman, white hair

Model units: about 30 high, facing +y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from . import paint as P
from . import sculpt as S
from .geometry import Part
from .units import Unit

# --------------------------------------------------------------------------- materials

def flesh(base, dark, seed=0, wounds=0.0, stitches=False) -> S.Material:
    """Dead skin: blotchy, darker in patches; `wounds` adds dark red-black gashes."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(look):
        p = look.p
        c = P.mix(base, dark, 0.15 + 0.6 * P.fbm(p, 2.2, seed) ** 1.5)
        if wounds:
            w = np.clip((P.fbm(p * np.array([1, 1, 2.0]), 1.6, seed + 31) - 0.7) * 6 * wounds, 0, 1)
            c = P.mix(c, P.hexrgb("#3a0e10"), w)
        if stitches:
            st = (np.abs(p[:, 0] - 0.3 * np.sin(p[:, 2] * 0.7)) < 0.12) & (np.abs(np.sin(p[:, 2] * 4.0)) > 0.6)
            c = np.where(st[:, None], P.hexrgb("#1e1a14"), c)
        return c
    return S.Material(color=color, shine=0.12, gloss=10, vivid=0.3)


def fur(base, dark, seed=0) -> S.Material:
    """Fur: streaks running down, darker at the roots."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(look):
        p = look.p
        streak = P.noise(p * np.array([3.0, 3.0, 0.5]), 1.0, seed)
        return P.mix(dark, base, np.clip(0.2 + 0.9 * streak * (0.6 + 0.4 * P.fbm(p, 3.0, seed + 1)), 0, 1))
    return S.Material(color=color, vivid=0.2)


def bandage(base=P.hexrgb("#ece0bc"), dark=P.hexrgb("#8a7e5c"), seed=0) -> S.Material:
    """Mummy wraps: bands wound round at a slant, dirty, with dark gaps between."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(look):
        p = look.p
        a = np.arctan2(p[:, 1], p[:, 0])
        band = (p[:, 2] * 1.6 + a * 0.6 + 0.4 * P.noise(p, 1.5, seed)) % 1.0
        c = P.mix(base, dark, 0.25 * P.fbm(p, 2.0, seed + 2) + 0.55 * np.clip((band - 0.82) * 6, 0, 1))
        return P.mix(c, P.hexrgb("#4a3a24"), np.clip((P.fbm(p, 3.0, seed + 5) - 0.6) * 3, 0, 1))
    return S.Material(color=color, vivid=0.6)


def cloth(base, dark=None, seed=0, torn=True) -> S.Material:
    base = np.asarray(base, float)
    dark = base * 0.55 if dark is None else np.asarray(dark, float)

    def color(look):
        p = look.p
        folds = 0.5 + 0.5 * np.sin(np.arctan2(p[:, 1], p[:, 0]) * 7 + P.noise(p, 2.0, seed) * 3)
        return P.mix(base, dark, 0.2 + 0.45 * folds * P.fbm(p, 2.5, seed + 1))
    return S.Material(color=color)


def team_cloth(lightness=0.6, seed=0) -> S.Material:
    def pattern(look):
        p = look.p
        folds = 0.82 + 0.18 * np.sin(np.arctan2(p[:, 1], p[:, 0]) * 7 + P.noise(p, 2.0, seed) * 3)
        return folds * (0.82 + 0.18 * P.noise(p, 1.5, seed + 11))
    return P.team(lightness, pattern)


def ghostly(base=P.hexrgb("#bfe8de"), seed=0) -> S.Material:
    """A ghost's stuff: pale, faintly shining by itself, wisps of light through it."""
    base = np.asarray(base, float)

    def color(look):
        return P.mix(base, base * 0.6, 0.5 * P.fbm(look.p, 2.5, seed))

    def glow(look):
        return 0.35 + 0.25 * P.noise(look.p, 2.0, seed + 3)
    return S.Material(color=color, glow=glow, vivid=0.6)


def chainmail(base=P.hexrgb("#7a7a84"), seed=0) -> S.Material:
    base = np.asarray(base, float)

    def color(look):
        p = look.p
        rings = (np.sin(p[:, 0] * 5.5 + p[:, 1] * 5.5) * np.sin(p[:, 2] * 5.5)) > 0.2
        c = np.where(rings[:, None], base, base * 0.45)
        return P.mix(c, P.RUST, np.clip((P.fbm(p, 2.0, seed) - 0.5) * 2.0, 0, 1))
    return S.Material(color=color, shine=0.5, gloss=20)


def eyes_material(base_m: S.Material, sockets: list, glow_rgb, pupil=0.45, dark=P.hexrgb("#120a0e")) -> S.Material:
    """A head with glowing eyes: `sockets` are (x, z, rx, rz) ellipses on its front (local coords), dark, with a
    glowing pupil in each."""
    glow_rgb = np.asarray(glow_rgb, float)

    def masks(p):
        sock = np.zeros(len(p), bool)
        pup = np.zeros(len(p), bool)
        for x, z, rx, rz in sockets:
            r = ((p[:, 0] - x) / rx) ** 2 + ((p[:, 2] - z) / rz) ** 2
            front = p[:, 1] > 0
            sock |= (r < 1) & front
            pup |= (r < pupil ** 2) & front
        return sock, pup

    def color(look):
        c = base_m.color(look) if callable(base_m.color) else np.broadcast_to(np.asarray(base_m.color, float),
                                                                             (len(look.p), 3)).copy()
        s, pu = masks(look.p)
        c = np.where(s[:, None], dark, c)
        return np.where(pu[:, None], glow_rgb, c)

    def glow(look):
        return masks(look.p)[1].astype(float)
    return S.Material(color=color, glow=glow, shine=base_m.shine, gloss=base_m.gloss, bump=base_m.bump,
                      vivid=base_m.vivid)


# --------------------------------------------------------------------------- bodies

@dataclass
class Body:
    kind: str = "skeleton"
    hip: float = 12.0
    shoulder: float = 22.0
    neck: float = 24.0
    arm_x: float = 4.6
    leg_x: float = 2.0
    hunch: float = 0.0  # degrees the torso leans forward
    arms_out: float = 0.0  # degrees the arms reach forward at rest (zombies, jiangshi)
    floating: bool = False  # no legs (ghosts, banshees)
    skin: Optional[S.Material] = None
    eye: tuple = (0.61, 1.0, 0.42)  # the eyes' glow colour
    size: float = 1.0  # the head's size


SKIN = {
    "zombie": lambda s: replace(flesh(P.hexrgb("#a2b886"), P.hexrgb("#5e6e48"), s, wounds=0.6, stitches=True),
                                vivid=0.7),
    "ghoul": lambda s: flesh(P.hexrgb("#8090a0"), P.hexrgb("#3e4656"), s, wounds=0.3),
    "imp": lambda s: flesh(P.hexrgb("#8a2a22"), P.hexrgb("#3a0e10"), s),
    "werewolf": lambda s: fur(P.hexrgb("#6a5a4a"), P.hexrgb("#2a221c"), s),
    "mummy": lambda s: bandage(seed=s),
    "vampire": lambda s: flesh(P.hexrgb("#d8d0d8"), P.hexrgb("#8a7a90"), s),
    "batman": lambda s: fur(P.hexrgb("#6a564c"), P.hexrgb("#2a1e1c"), s),
    "draugr": lambda s: flesh(P.hexrgb("#6e7c86"), P.hexrgb("#2e363e"), s, wounds=0.4),
    "jiangshi": lambda s: flesh(P.hexrgb("#a8b8a4"), P.hexrgb("#5a6858"), s),
    "ghost": lambda s: ghostly(seed=s),
    "banshee": lambda s: ghostly(P.hexrgb("#d6e4ee"), seed=s),
}
EYES = {"skeleton": (0.61, 1.0, 0.42), "zombie": (1.0, 0.85, 0.3), "ghoul": (0.75, 0.95, 1.0),
        "imp": (1.0, 0.75, 0.15), "werewolf": (1.0, 0.7, 0.1), "mummy": (0.55, 1.0, 0.45), "vampire": (1.0, 0.15, 0.1),
        "batman": (1.0, 0.3, 0.15), "calavera": (1.0, 0.6, 0.15), "ghost": (0.4, 1.0, 0.95),
        "jiangshi": (0.9, 0.95, 0.85), "knight": (1.0, 0.12, 0.05), "draugr": (0.45, 0.8, 1.0),
        "banshee": (0.85, 0.95, 1.0)}


def body(kind: str, seed: int = 0, **kw) -> Body:
    b = Body(kind=kind, eye=EYES.get(kind, EYES["skeleton"]), **kw)
    if b.skin is None:
        b.skin = SKIN[kind](seed) if kind in SKIN else P.bone(seed)
    if kind in ("zombie", "ghoul") and "hunch" not in kw:
        b.hunch = 12 if kind == "zombie" else 18
    if kind in ("ghost", "banshee"):
        b.floating = True
    return b


def _skull_shapes(b: Body, seed: int) -> list:
    """A skull: cranium with deep glowing sockets, a nose hole, a jaw with teeth."""
    from .undead import _jaw_material, _skull_material
    n = b.neck
    k = b.size
    skull_m = _skull_material(b.eye, seed)
    if b.kind == "calavera":
        skull_m = _calavera_material(b.eye, seed)
    return [S.Ellipsoid(skull_m, S._orient((0, 0.4, n + 3.6 * k)), radii=(3.4 * k, 3.6 * k, 3.3 * k)),
            S.Ellipsoid(_jaw_material(seed), S._orient((0, 1.7 * k, n + 1.0 * k)), radii=(2.3 * k, 2.1 * k,
                                                                                          1.25 * k))]


def _calavera_material(glow, seed):
    """A sugar skull: white bone painted round the eyes with petals, a heart on the nose, dots on the brow."""
    from .undead import _skull_material
    base = _skull_material(glow, seed)
    pinks = [P.hexrgb("#e0407a"), P.hexrgb("#3a8ad8"), P.hexrgb("#f0c020"), P.hexrgb("#38b878")]

    def color(look):
        c = base.color(look)
        p = look.p
        front = p[:, 1] > 0.8
        for k, sx in enumerate((-1, 1)):
            dx, dz = p[:, 0] - sx * 1.3, p[:, 2] + 0.1
            r = np.hypot(dx, dz)
            ang = np.arctan2(dz, dx)
            petal = front & (r > 1.15) & (r < 1.15 + 0.55 * (0.5 + 0.5 * np.cos(ang * 6))) & (r < 1.75)
            c = np.where(petal[:, None], pinks[k], c)
        dots = front & (p[:, 2] > 1.4) & (np.hypot((p[:, 0] % 1.1) - 0.55, (p[:, 2] % 1.1) - 0.55) < 0.22)
        c = np.where(dots[:, None], pinks[2], c)
        heart = front & (np.abs(p[:, 0]) < 0.6) & (p[:, 2] < -0.9) & (p[:, 2] > -2.0)
        return np.where(heart[:, None], pinks[0], c)
    return S.Material(color=color, glow=base.glow, shine=0.3, gloss=16, vivid=0.9)


def _head_shapes(b: Body, seed: int) -> list:
    n, k = b.neck, b.size
    kind = b.kind
    eye = b.eye
    if kind in ("skeleton", "calavera"):
        return _skull_shapes(b, seed)
    if kind == "werewolf":
        m = eyes_material(b.skin, [(-1.4, 0.8, 0.6, 0.45), (1.4, 0.8, 0.6, 0.45)], eye, 0.6)
        return [S.Ellipsoid(m, S._orient((0, 0.2, n + 3.0)), radii=(3.2, 3.3, 3.0)),
                S.rod((0, 1.6, n + 2.0), (0, 6.2, n + 1.4), 1.9, b.skin, 1.0),  # the snout
                S.ball((0, 6.4, n + 1.6), 0.7, S.mat(P.hexrgb("#141012"))),
                S.hull([(-0.9, 5.0, n + 0.8), (0.9, 5.0, n + 0.8), (0, 5.4, n - 0.4)], S.mat(P.BONE)),
                S.hull([(-2.6, -0.2, n + 4.6), (-1.0, -0.4, n + 4.8), (-1.9, 0.2, n + 8.2)], b.skin),
                S.hull([(2.6, -0.2, n + 4.6), (1.0, -0.4, n + 4.8), (1.9, 0.2, n + 8.2)], b.skin)]
    if kind == "batman":
        m = eyes_material(b.skin, [(-1.2, 0.6, 0.55, 0.5), (1.2, 0.6, 0.55, 0.5)], eye, 0.6)
        return [S.Ellipsoid(m, S._orient((0, 0.3, n + 3.0)), radii=(3.0, 3.0, 3.0)),
                S.ball((0, 2.9, n + 2.2), (1.2, 0.8, 0.8), S.mat(P.hexrgb("#2a1c20"))),
                S.hull([(-2.6, 0.0, n + 4.0), (-0.8, 0.0, n + 5.0), (-3.4, -0.3, n + 9.0), (-2.0, 0.4, n + 6.0)],
                       b.skin),
                S.hull([(2.6, 0.0, n + 4.0), (0.8, 0.0, n + 5.0), (3.4, -0.3, n + 9.0), (2.0, 0.4, n + 6.0)],
                       b.skin),
                S.hull([(-0.9, 2.6, n + 1.2), (-0.5, 2.6, n + 1.2), (-0.7, 2.7, n + 0.0)], S.mat(P.BONE)),
                S.hull([(0.9, 2.6, n + 1.2), (0.5, 2.6, n + 1.2), (0.7, 2.7, n + 0.0)], S.mat(P.BONE))]
    if kind == "knight":  # a black great helm with a glowing visor slit and spikes
        plate = P.metal(P.hexrgb("#2a262e"), rust=0.3, seed=seed, shine=0.7)

        def visor(look):
            p = look.p
            return ((np.abs(p[:, 2] - 0.4) < 0.35) & (p[:, 1] > 1.5) & (np.abs(p[:, 0]) < 2.6)).astype(float)
        helm_m = S.Material(color=lambda l: np.where(visor(l)[:, None] > 0, np.asarray(eye, float),
                                                     plate.color(l)), glow=visor, shine=0.7, gloss=30)
        return [S.rod((0, 0.3, n - 0.2), (0, 0.3, n + 6.2), 3.6, helm_m, 3.2),
                S.ball((0, 0.3, n + 6.2), (3.2, 3.2, 1.4), plate),
                S.hull([(-0.6, 0.3, n + 7.2), (0.6, 0.3, n + 7.2), (0, -0.3, n + 7.2), (0, 0.9, n + 7.2),
                        (0, 0.3, n + 10.0)], plate),
                S.hull([(-3.3, 0.3, n + 4.8), (-3.3, 0.3, n + 6.0), (-3.3, -0.4, n + 5.4), (-6.0, 0.3, n + 7.6)],
                       plate),
                S.hull([(3.3, 0.3, n + 4.8), (3.3, 0.3, n + 6.0), (3.3, -0.4, n + 5.4), (6.0, 0.3, n + 7.6)],
                       plate)]
    # flesh heads: a slightly long skull shape with sunken glowing eyes and a gaping mouth
    eyes = [(-1.25, 0.6, 0.62, 0.48), (1.25, 0.6, 0.62, 0.48)]
    m = eyes_material(b.skin, eyes, eye, 0.55)
    shapes = [S.Ellipsoid(m, S._orient((0, 0.3, n + 3.3 * k)), radii=(3.0 * k, 3.2 * k, 3.5 * k))]
    if kind in ("zombie", "ghoul", "draugr", "banshee", "imp"):  # a gaping mouth
        shapes.append(S.Ellipsoid(S.mat(P.hexrgb("#1a0a0c")), S._orient((0, 2.6 * k, n + 1.6 * k)),
                                  radii=(1.1 * k, 0.6 * k, 0.9 * k)))
    if kind == "zombie":
        shapes.append(S.ball((-1.6, 0.0, n + 6.0), (1.0, 1.4, 0.4), P.mottled(P.hexrgb("#2a2018"))))  # matted hair
    if kind == "imp":
        horn = S.mat(P.hexrgb("#1a1414"), shine=0.4)
        shapes += [S.rod((-1.6, 0.4, n + 5.6), (-2.8, -0.8, n + 9.0), 0.7, horn, 0.1),
                   S.rod((1.6, 0.4, n + 5.6), (2.8, -0.8, n + 9.0), 0.7, horn, 0.1)]
    if kind == "vampire":
        hair = S.mat(P.hexrgb("#141016"), shine=0.6, gloss=30)
        shapes += [S.ball((0, -0.5, n + 4.6), (3.15, 3.2, 2.6), hair).clip((0, 1, -0.6), 0.9),
                   S.hull([(-0.7, 2.9, n + 1.4), (-0.4, 2.9, n + 1.4), (-0.55, 3.0, n + 0.6)], S.mat(P.BONE)),
                   S.hull([(0.7, 2.9, n + 1.4), (0.4, 2.9, n + 1.4), (0.55, 3.0, n + 0.6)], S.mat(P.BONE))]
    if kind == "jiangshi":  # the Qing hat and the yellow paper talisman on the forehead
        hat = S.mat(P.hexrgb("#1a1418"))
        shapes += [S.rod((0, 0.3, n + 5.6), (0, 0.3, n + 6.6), 3.6, hat, 3.1),
                   S.ball((0, 0.3, n + 6.6), (3.1, 3.1, 1.6), hat),
                   S.ball((0, 0.3, n + 8.0), 0.6, P.team(0.6)),
                   S.block((-1.0, 3.3, n + 0.6), (1.0, 3.6, n + 5.2),
                           S.Material(color=_talisman, vivid=0.8))]
    if kind == "mummy":
        shapes.append(S.ball((0, 0.3, n + 3.3), (3.15, 3.35, 3.65), b.skin).clip((0, -1, 0), -2.2))
    if kind == "banshee":  # long white hair streaming down the back
        hair = ghostly(P.hexrgb("#f2f4f6"), seed + 3)
        shapes += [S.ball((0, -0.6, n + 4.0), (3.4, 3.4, 3.2), hair).clip((0, 1, 0), 0.8),
                   S.rod((0, -2.2, n + 4.0), (0, -3.4, n - 9.0), 2.8, hair, 0.8)]
    return shapes


def _talisman(look):
    p = look.p
    c = np.broadcast_to(P.hexrgb("#e8d060"), (len(p), 3)).copy()
    mark = (np.abs(np.sin(p[:, 2] * 3.0)) < 0.3) & (np.abs(p[:, 0]) < 0.4)
    return np.where(mark[:, None], P.hexrgb("#b01810"), c)


def _limb_shapes(b: Body, a, c, r0, r1) -> list:
    if b.kind in ("skeleton", "calavera"):
        r0, r1 = r0 * 0.75, r1 * 0.75
        return [S.rod(a, c, r0, b.skin, r1), S.ball(a, r0 * 1.45, b.skin), S.ball(c, r1 * 1.45, b.skin)]
    return S.limb(a, c, r0, b.skin, r1)


@dataclass
class Gear:
    helmet: Optional[str] = None  # kettle, nasal, great, horned, hood, cowl, crown, turban, fez, feathers, pelt, oni
    cloth: Optional[str] = None  # loin, tunic, tabard, robe, rags, dress, wraps
    cape: bool = False
    armour: Optional[str] = None  # plate, mail, lamellar
    shield: Optional[str] = None  # round, kite, oval
    right: Optional[str] = None  # what the right hand holds (see WEAPONS)
    left: Optional[str] = None
    chest: Optional[str] = None  # crossbow, gun, blunderbuss (two-handed, held at the chest)
    carry: Optional[str] = None  # what a villager carries home (wood, gold, stone, food, meat, fish, berries)
    back: Optional[str] = None  # quiver, wings, banner, keg


def _armour_shapes(b: Body, g: Gear, seed: int) -> list:
    sh = b.shoulder
    out = []
    if g.armour == "plate":
        plate = P.metal(P.hexrgb("#3a363e") if b.kind == "knight" else P.hexrgb("#8e8e98"), rust=0.6, seed=seed)
        out += [S.ball((0, 0.4, sh - 3.3), (4.0, 3.0, 4.2), plate).clip((0, 0, -1), 3.6),
                S.ball((b.arm_x + 0.3, 0, sh + 0.2), (2.0, 2.2, 1.5), plate),
                S.ball((-b.arm_x - 0.3, 0, sh + 0.2), (2.0, 2.2, 1.5), plate)]
        if b.kind == "knight":
            for x in (b.arm_x + 0.6, -b.arm_x - 0.6):
                out.append(S.hull([(x - 0.6, -0.5, sh + 1.2), (x + 0.6, -0.5, sh + 1.2), (x, 0.5, sh + 1.2),
                                   (x * 1.15, 0, sh + 3.6)], plate))
    elif g.armour == "mail":
        out.append(S.ball((0, 0.3, sh - 3.6), (3.9, 2.9, 4.6), chainmail(seed=seed)).clip((0, 0, -1), 4.4))
    elif g.armour == "lamellar":
        def plates(look):
            p = look.p
            rows = (p[:, 2] * 1.6) % 1.0 < 0.18
            return np.where(rows, 0.35, 0.72)
        out.append(S.ball((0, 0.3, sh - 3.6), (4.0, 3.0, 4.6), P.team(0.7, plates)).clip((0, 0, -1), 4.4))
        for x in (b.arm_x + 0.4, -b.arm_x - 0.4):
            out.append(S.block((x - 1.6, -1.6, sh - 3.5), (x + 1.6, 1.6, sh + 0.6), P.team(0.6, plates),
                               rot=(0, 18 if x > 0 else -18, 0)))
    return out


DRAB = [P.hexrgb("#80684e"), P.hexrgb("#6a6a74"), P.hexrgb("#6c7250"), P.hexrgb("#74504a"), P.hexrgb("#5e5868")]


def _torn_skirt(top, bottom, r0, r1, m) -> S.Shape:
    """A short skirt hanging from the waist, its hem torn (cut by tilted planes)."""
    return S.rod((0, 0.1, top), (0, 0.2, bottom), r0, m, r1, capped=False)


def _chest(b: Body, m, grow=0.35, low=None) -> S.Shape:
    """Cloth or armour fitting over the chest: the body's chest, a little bigger, cut off at `low`."""
    sh, h = b.shoulder, b.hip
    r = {"werewolf": (4.4, 3.2, 5.0), "imp": (3.2, 2.4, 3.6), "ghoul": (3.2, 2.4, 4.4)}.get(b.kind, (3.6, 2.6, 4.4))
    shape = S.ball((0, 0.2, sh - 3.6), (r[0] + grow, r[1] + grow, r[2] + grow), m)
    return shape.clip((0, 0, -1), -(low if low is not None else h + 0.5) + (sh - 3.6))


def _belt(b: Body, z, team=True, seed=0) -> S.Shape:
    m = team_cloth(0.62, seed) if team else P.wood(P.hexrgb("#3a2618"), P.hexrgb("#1e120a"), along=0)
    return S.torus((0, 0.2, z), 3.2, 0.55, m)


def _cloth_shapes(b: Body, g: Gear, seed: int) -> list:
    """Clothes: drab colours fitted to the body, the owner's colour only on sashes, belts, tabards and trim (as the
    game's own units carry it), so the creatures inside still show."""
    h, sh = b.hip, b.shoulder
    drab = cloth(DRAB[seed % len(DRAB)], seed=seed)
    if g.cloth == "loin":
        return [_torn_skirt(h + 1.0, h - 3.6, 3.1, 3.5, team_cloth(0.55, seed)), _belt(b, h + 1.0, False)]
    if g.cloth == "tunic":
        if b.kind in ("skeleton", "calavera"):  # no shirt over the ribs: a sash across them instead
            sash = S.rod((-3.4, 2.4, sh + 0.4), (3.0, 2.6, h + 1.2), 0.7, team_cloth(0.62, seed))
            back = S.rod((-3.4, -2.6, sh + 0.4), (3.0, -2.4, h + 1.2), 0.7, team_cloth(0.5, seed))
            return [sash, back, _torn_skirt(h + 1.2, h - 3.2, 3.2, 3.7, drab), _belt(b, h + 1.2, False)]
        return [_chest(b, drab), _torn_skirt(h + 1.2, h - 3.2, 3.4, 3.9, drab), _belt(b, h + 1.2, True, seed)]
    if g.cloth == "tabard":  # narrow panels front and back over the chest, in the owner's colour, a skull on it
        def emblem(look):
            p = look.p
            skull = (np.hypot(p[:, 0] / 1.0, (p[:, 2] - (sh - 4.5)) / 1.1) < 1.0)
            return np.where(skull, 0.25, 1.0)
        t = P.team(0.6, emblem)
        return [S.block((-2.3, 2.4, h - 4.0), (2.3, 2.8, sh - 0.6), t),
                S.block((-2.3, -3.1, h - 4.0), (2.3, -2.7, sh - 0.6), team_cloth(0.5, seed)),
                _belt(b, h + 1.2, False)]
    if g.cloth == "robe":  # a long dark robe, the owner's colour in the trim and the sash
        dark = cloth(P.hexrgb("#24202a"), seed=seed)
        return [S.rod((0, 0, sh + 0.3), (0, 0, 0.8), 3.0, dark, 4.8),
                S.rod((0, 0, 1.6), (0, 0, 0.6), 4.65, team_cloth(0.5, seed), 4.85, capped=False),
                _belt(b, h + 1.4, True, seed)]
    if g.cloth == "rags":  # a torn shirt and a ragged skirt, a rag tied round the arm in the owner's colour
        return [_chest(b, drab, 0.25), _torn_skirt(h + 1.0, h - 5.0, 3.2, 3.7, cloth(P.hexrgb("#3e382e"), seed=seed)),
                _belt(b, h + 1.0, True, seed)]
    if g.cloth == "dress":
        gown = ghostly(P.hexrgb("#c8d8e0"), seed + 2) if b.kind in ("ghost", "banshee") else drab
        return [S.rod((0, 0, sh + 0.2), (0, 0, -1.0), 3.0, gown, 6.4), _belt(b, h + 2.0, True, seed)]
    if g.cloth == "wraps":  # a long Qing robe, dark, a gold hem, the owner's colour in the sash
        dark = cloth(P.hexrgb("#38586a"), seed=seed)
        return [S.rod((0, 0, sh + 0.5), (0, 0, 0.8), 3.4, dark, 4.6),
                S.rod((0, 0, 1.8), (0, 0, 0.6), 4.45, S.mat(P.GOLD, shine=0.6), 4.65, capped=False),
                _belt(b, h + 1.2, True, seed)]
    return []


def _helmet_shapes(b: Body, kind: str, seed: int) -> list:
    n = b.neck
    top = n + 3.6 * b.size
    iron = P.metal(P.hexrgb("#9a9aa6"), rust=0.5, seed=seed + 3)
    dark_iron = P.metal(P.hexrgb("#4a4650"), rust=0.4, seed=seed + 3)
    if kind == "kettle":
        return [S.ball((0, 0.3, top + 0.4), (3.7, 3.9, 2.8), iron).clip((0, 0, -1), -0.2),
                S.rod((0, 0.3, top + 0.1), (0, 0.3, top + 0.5), 5.4, iron)]
    if kind == "nasal":
        return [S.ball((0, 0.3, top + 0.2), (3.75, 3.95, 3.6), iron).clip((0, 0, -1), 0.3),
                S.block((-0.35, 3.8, top - 2.6), (0.35, 4.4, top + 0.6), iron)]
    if kind == "great":
        return [S.rod((0, 0.3, top - 4.2), (0, 0.3, top + 1.6), 3.9, iron, 3.6),
                S.block((-2.6, 3.7, top - 1.6), (2.6, 4.4, top - 1.0), S.mat(P.hexrgb("#120c10")))]
    if kind == "horned":
        horn = S.mat(P.hexrgb("#d8ccb0"), shine=0.3)
        return [S.rod((0, 0.3, top - 4.2), (0, 0.3, top + 1.6), 3.9, dark_iron, 3.6),
                S.block((-2.6, 3.7, top - 1.6), (2.6, 4.4, top - 1.0), S.mat(P.hexrgb("#120c10"))),
                S.rod((3.4, 0.3, top - 0.4), (6.0, 0.8, top + 3.4), 0.9, horn, 0.15),
                S.rod((-3.4, 0.3, top - 0.4), (-6.0, 0.8, top + 3.4), 0.9, horn, 0.15)]
    if kind in ("hood", "cowl"):  # a hood over the head, open at the face, and a short cape over the shoulders
        hood = team_cloth(0.42, seed) if kind == "hood" else cloth(P.hexrgb("#1c181e"), seed=seed)
        mantle = cloth(DRAB[seed % len(DRAB)], seed=seed) if kind == "hood" else hood
        return [S.ball((0, -0.3, top - 0.8), (3.9, 4.1, 4.3), hood).clip((0, 1, 0), 1.9),
                S.rod((0, -0.2, b.shoulder + 1.6), (0, -0.2, b.shoulder - 1.4), 2.6, mantle, 5.0)]
    if kind == "crown":
        gold = P.metal(P.GOLD, seed=seed, shine=0.8)
        out = [S.rod((0, 0.3, top + 0.2), (0, 0.3, top + 1.6), 3.3, gold, 3.4, capped=False)]
        for a in range(0, 360, 60):
            r = math.radians(a)
            out.append(S.hull([(math.cos(r) * 3.0 - 0.6, math.sin(r) * 3.0 + 0.3, top + 1.5),
                               (math.cos(r) * 3.0 + 0.6, math.sin(r) * 3.0 + 0.3, top + 1.5),
                               (math.cos(r) * 3.3, math.sin(r) * 3.3 + 0.3, top + 3.2)], gold))
        out.append(S.ball((0, 3.5, top + 1.0), 0.5, S.mat(P.BLOOD, glow=0.5)))
        return out
    if kind == "turban":
        return [S.torus((0, 0.2, top - 0.6), 3.2, 1.3, team_cloth(0.65, seed)),
                S.ball((0, 0.2, top + 0.4), (3.0, 3.0, 2.2), team_cloth(0.6, seed))]
    if kind == "fez":
        return [S.rod((0, 0.3, top - 0.6), (0, 0.3, top + 2.4), 2.6, P.team(0.55), 2.1),
                S.rod((0, 0.3, top + 2.4), (1.6, -0.6, top + 0.4), 0.2, S.mat(P.GOLD))]
    if kind == "feathers":  # marigolds and plumes (Day of the Dead)
        out = []
        cols = [P.hexrgb("#f08a10"), P.hexrgb("#e0407a"), P.hexrgb("#f0c020")]
        for k, a in enumerate(range(-70, 71, 20)):
            r = math.radians(a)
            base = (math.sin(r) * 2.6, math.cos(r) * 0.6 - 1.0, top + 1.0)
            tip = (math.sin(r) * 6.0, -2.0, top + 7.5 - abs(a) * 0.04)
            out.append(S.Ellipsoid(P.team(0.6) if k % 3 == 1 else S.mat(cols[k % 3], vivid=1.0),
                                   S.axis_frame(base, tip) @ S.affine(None, (0, 0, 3.6)),
                                   radii=(0.9, 0.35, 3.8)))
        out.append(S.torus((0, 0.3, top - 0.6), 3.3, 0.8, S.mat(P.hexrgb("#f08a10"), vivid=1.0)))
        return out
    if kind == "pelt":  # a jaguar's head over the warrior's
        spots = S.Material(color=lambda l: np.where(
            (P.noise(l.p, 1.0, 5) > 0.62)[:, None], P.hexrgb("#1a1410"), P.hexrgb("#c08a30")), vivid=0.6)
        return [S.ball((0, -0.3, top + 0.2), (3.9, 4.3, 3.4), spots).clip((0, 1, 0.3), 1.6),
                S.ball((0, 2.6, top + 1.4), (2.6, 2.2, 1.6), spots),
                S.hull([(-1.0, 3.8, top + 0.4), (-0.6, 3.8, top + 0.4), (-0.8, 4.0, top - 0.8)], S.mat(P.BONE)),
                S.hull([(1.0, 3.8, top + 0.4), (0.6, 3.8, top + 0.4), (0.8, 4.0, top - 0.8)], S.mat(P.BONE))]
    if kind == "oni":  # a red demon mask with horns over a kabuto
        red = S.mat(P.hexrgb("#a01a14"), shine=0.6, gloss=24, vivid=0.7)
        lac = P.metal(P.hexrgb("#1c181c"), seed=seed, shine=0.8)
        return [S.ball((0, 0.2, top + 0.2), (3.9, 4.1, 3.4), lac).clip((0, 0, -1), -0.4),
                S.rod((0, 0.2, top - 0.6), (0, -0.6, top - 3.2), 4.8, lac, 5.6, capped=False),
                S.ball((0, 3.1, top - 3.2), (2.6, 1.0, 2.4), red),
                S.hull([(-1.2, 3.7, top - 4.6), (1.2, 3.7, top - 4.6), (0, 3.9, top - 5.4)], S.mat(P.BONE)),
                S.rod((-2.0, 0.6, top + 2.4), (-3.4, 1.2, top + 6.6), 0.7, S.mat(P.GOLD, shine=0.8), 0.15),
                S.rod((2.0, 0.6, top + 2.4), (3.4, 1.2, top + 6.6), 0.7, S.mat(P.GOLD, shine=0.8), 0.15)]
    if kind == "viking":
        return [S.ball((0, 0.3, top + 0.2), (3.75, 3.95, 3.6), dark_iron).clip((0, 0, -1), 0.3),
                S.block((-2.4, 3.8, top - 2.2), (2.4, 4.3, top - 1.2), dark_iron)]
    if kind == "tricorn":
        hat = S.mat(P.hexrgb("#1a1618"))
        return [S.rod((0, 0.3, top - 0.2), (0, 0.3, top + 0.3), 5.6, hat), S.ball((0, 0.3, top + 0.3),
                                                                                (3.2, 3.2, 2.2), hat),
                S.ball((0, 0.3, top + 0.35), (5.4, 5.4, 0.6), P.team(0.5)).clip((0, 0, -1), -0.2)]
    return []


# --------------------------------------------------------------------------- what they hold

def _forward(up_deg=20.0):
    a = math.radians(up_deg)
    return np.array([0.0, math.cos(a), math.sin(a)]), np.array([0.0, -math.sin(a), math.cos(a)])


def _blade(length=11.0, width=0.85, m=None, up_deg=20.0, start=1.6) -> list:
    d, up = _forward(up_deg)
    m = m or P.metal(P.hexrgb("#9696a2"), rust=0.6)
    base = d * start
    tip = base + d * length
    t = 0.22
    return [S.hull([base + up * width + [t, 0, 0], base - up * width + [t, 0, 0], base + up * width - [t, 0, 0],
                    base - up * width - [t, 0, 0], tip - d * 1.8 + up * width * 0.8, tip - d * 1.8 - up * width * 0.8,
                    tip], m)]


def _hilt(up_deg=20.0, guard=2.4) -> list:
    d, _ = _forward(up_deg)
    hilt = P.metal(P.hexrgb("#5a4a3a"), rust=0.4)
    return [S.block((-0.5, -0.35, -guard), (0.5, 0.35, guard), hilt, rot=(up_deg, 0, 0), at=tuple(d * 1.6)),
            S.rod(-d * 1.4, d * 1.6, 0.45, P.wood(P.hexrgb("#4a2c1c"), P.hexrgb("#2a1810"))),
            S.ball(-d * 1.6, 0.7, hilt)]


def _shaft(length, back=4.0, up_deg=20.0, r=0.42, m=None) -> list:
    d, _ = _forward(up_deg)
    m = m or P.wood(P.hexrgb("#4e3626"), P.hexrgb("#2a1a12"), along=2)
    return [S.rod(-d * back, d * (length - back), r, m)], d


def weapon(name: str, seed: int = 0) -> list:
    """What a hand holds, with the grip at the origin, pointing forward (+y) and a little up."""
    steel = P.metal(P.hexrgb("#9696a2"), rust=0.6, seed=seed)
    wood = P.wood(P.hexrgb("#4e3626"), P.hexrgb("#2a1a12"), along=2)
    if name == "sword":
        return _blade(11.0) + _hilt()
    if name == "longsword":
        return _blade(14.0, 0.8) + _hilt(guard=2.8)
    if name == "greatsword":
        return _blade(18.0, 1.1) + _hilt(guard=3.4)
    if name == "runesword":  # a black blade with green glowing runes
        def runes(look):
            p = look.p
            return ((np.abs(np.sin(p[:, 1] * 3.0)) > 0.8) & (np.abs(p[:, 0]) < 0.3)).astype(float)
        m = S.Material(color=lambda l: np.where(runes(l)[:, None] > 0, P.SLIME, P.hexrgb("#2a2830")),
                       glow=runes, shine=0.8, gloss=30)
        return _blade(17.0, 1.1, m) + _hilt(guard=3.4)
    if name == "katana":
        return _blade(14.0, 0.55, P.metal(P.hexrgb("#c8ccd4"), seed=seed, shine=0.9), up_deg=28) + \
            [S.rod(_forward(28)[0] * -2.2, _forward(28)[0] * 1.4, 0.45, P.team(0.3))]
    if name in ("spear", "pike", "javelin"):
        L = {"spear": 18.0, "pike": 26.0, "javelin": 13.0}[name]
        shapes, d = _shaft(L, back=4.0 if name != "javelin" else 2.0)
        tip = d * (L - (4.0 if name != "javelin" else 2.0))
        side = np.array([1.0, 0, 0])
        up = np.cross(d, side)
        shapes.append(S.hull([tip + up * 0.8, tip - up * 0.8, tip + side * 0.2, tip - side * 0.2, tip + d * 3.2],
                             steel))
        return shapes
    if name == "halberd":
        shapes, d = _shaft(22.0)
        tip = d * 18.0
        up = np.array([0.0, -d[2], d[1]])
        shapes.append(S.hull([tip + d * 0.0, tip + d * 3.0, tip + d * 1.5 + up * 0.5, tip + up * 0.0], steel))
        shapes.append(S.hull([tip - d * 2.5 + [0.2, 0, 0], tip - d * 2.5 - [0.2, 0, 0], tip + d * 0.4 + [0.2, 0, 0],
                              tip + d * 0.4 - [0.2, 0, 0], tip - d * 3.2 - up * 3.2, tip + d * 1.2 - up * 3.6],
                             steel))
        shapes.append(S.hull([tip + d * 2.0, tip + d * 2.0 + [0.15, 0, 0], tip + d * 5.0], steel))
        return shapes
    if name in ("axe", "bigaxe"):
        big = name == "bigaxe"
        shapes, d = _shaft(11.0 if not big else 15.0, back=1.5)
        head = d * (8.5 if not big else 12.0)
        up = np.array([0.0, -d[2], d[1]])
        w = 2.6 if not big else 3.6
        shapes.append(S.hull([head + d * 1.2 + [0.25, 0, 0], head + d * 1.2 - [0.25, 0, 0], head - d * 1.2 + [0.25, 0, 0],
                              head - d * 1.2 - [0.25, 0, 0], head + up * w + d * 1.8, head + up * w - d * 1.8], steel))
        return shapes
    if name == "club":  # an obsidian-edged war club (macuahuitl)
        shapes, d = _shaft(12.0, back=1.5, r=0.6)
        up = np.array([0.0, -d[2], d[1]])
        obs = S.mat(P.hexrgb("#14101a"), shine=0.9, gloss=40)
        for k in range(5):
            c = d * (5.0 + k * 1.4)
            for s in (1, -1):
                shapes.append(S.hull([c + up * 0.4 * s, c + d * 0.8 + up * 0.4 * s, c + d * 0.4 + up * 1.4 * s,
                                      c + [0.15, 0, 0], c - [0.15, 0, 0]], obs))
        return shapes
    if name == "scythe":
        shapes, d = _shaft(20.0, back=6.0, up_deg=70)
        top = d * 14.0
        blade = []
        for k in range(6):
            a = k / 5
            blade.append(top + np.array([0, 1.0, 0]) * (a * 8.0) + np.array([0, 0, -1.0]) * (a * a * 3.0))
        for a, b in zip(blade, blade[1:]):
            shapes.append(S.rod(a, b, 0.5, steel, 0.4))
        return shapes
    if name in ("pickaxe", "hammer", "hoe", "woodaxe"):
        shapes, d = _shaft(10.0, back=1.2)
        head = d * 8.0
        up = np.array([0.0, -d[2], d[1]])
        if name == "pickaxe":
            shapes += [S.rod(head - up * 0.0, head + up * 3.2 + d * 0.8, 0.45, steel, 0.12),
                       S.rod(head, head - up * 3.2 + d * 0.8, 0.45, steel, 0.12)]
        elif name == "hammer":
            shapes.append(S.block((-1.0, -1.0, -2.0), (1.0, 1.0, 2.0), steel, at=tuple(head)))
        elif name == "hoe":
            shapes.append(S.block((-1.2, 0, -0.15), (1.2, 2.4, 0.15), steel, rot=(-70, 0, 0), at=tuple(head)))
        else:
            shapes.append(S.hull([head + up * 0.4, head - up * 0.4, head + up * 2.8 + d * 1.4,
                                  head + up * 2.8 - d * 1.4, head + [0.25, 0, 0], head - [0.25, 0, 0]], steel))
        return shapes
    if name in ("bow", "longbow", "bonebow"):
        L = 7.5 if name != "longbow" else 10.0
        m = P.wood(P.hexrgb("#5a3e26"), P.hexrgb("#2a1a10"), along=2) if name != "bonebow" else P.bone(seed)
        pts = []
        for k in range(9):
            a = (k / 8 - 0.5) * 2
            pts.append(np.array([0.0, 1.0 * (1 - a * a) + 0.4, a * L]))
        shapes = []
        for a, b in zip(pts, pts[1:]):
            shapes.append(S.rod(a, b, 0.42 - 0.12 * abs(a[2]) / L, m))
        string = S.mat(P.hexrgb("#d8d0c0"))
        shapes.append(S.rod(pts[0], pts[-1], 0.08, string))
        return shapes
    if name == "staff":  # a necromancer's staff, a skull on top with glowing eyes and a green flame
        from .graveyard import skull
        shapes, d = _shaft(20.0, back=6.0, up_deg=88, r=0.45, m=P.wood(P.hexrgb("#3a2a22"), P.hexrgb("#160e0a"),
                                                                         along=2))
        top = d * 14.0
        shapes += [_moved(s, top - (0, 0, 0.6)) for s in skull((0, 0, 0), 1.5, seed)]
        flame = S.Ellipsoid(S.mat(P.SLIME, glow=1.0), S._orient(top + (0, 0, 4.2)), radii=(0.9, 0.9, 1.8))
        flame.casts = False
        shapes.append(flame)
        return shapes
    if name == "lantern":
        frame = P.metal(P.hexrgb("#3a3430"), rust=0.4, seed=seed)
        return [S.rod((0, 0.6, 0.0), (0, 0.6, -2.0), 0.12, frame),
                S.block((-1.0, -0.4, -4.6), (1.0, 1.6, -2.0), S.mat(P.hexrgb("#ffb347"), glow=0.95)),
                S.block((-1.2, -0.6, -2.2), (1.2, 1.8, -1.8), frame),
                S.block((-1.2, -0.6, -4.9), (1.2, 1.8, -4.5), frame)]
    if name == "rod":  # a fishing rod
        shapes, d = _shaft(16.0, back=2.0, up_deg=40, r=0.25)
        tip = d * 14.0
        shapes.append(S.rod(tip, tip + (0, 2.5, -9.0), 0.06, S.mat(P.hexrgb("#c8c0b0"))))
        return shapes
    if name == "basket":
        return [S.rod((0, 1.6, -3.0), (0, 1.6, -0.2), 2.0, P.wood(P.hexrgb("#8a6a3a"), P.hexrgb("#4a3418"),
                                                                    along=0, scale=0.3), 2.4, capped=True)]
    if name == "torch":
        shapes, d = _shaft(8.0, back=1.0, up_deg=70, r=0.4)
        top = d * 7.0
        f = S.Ellipsoid(S.mat(P.hexrgb("#ffb030"), glow=1.0), S._orient(top + (0, 0, 1.4)), radii=(0.9, 0.9, 1.9))
        f.casts = False
        return shapes + [f]
    if name == "orb":
        o = S.ball((0, 1.2, 0.8), 1.4, S.mat(P.SLIME, glow=0.9))
        o.casts = False
        return [o]
    if name == "claws":
        return []
    raise KeyError(name)


def chest_weapon(name: str, seed: int = 0) -> list:
    """A two-handed weapon held across the chest (part "item_c"), pointing forward from the origin."""
    wood = P.wood(P.hexrgb("#4a3020"), P.hexrgb("#24160e"), along=1)
    steel = P.metal(P.hexrgb("#6a6a74"), rust=0.5, seed=seed)
    if name in ("crossbow", "repeater"):
        out = [S.block((-0.6, -3.0, -0.6), (0.6, 6.0, 0.6), wood)]
        pts = [np.array([x, 5.4 - 0.08 * x * x, 0.0]) for x in np.linspace(-5, 5, 7)]
        for a, b in zip(pts, pts[1:]):
            out.append(S.rod(a, b, 0.3, steel))
        out.append(S.rod(pts[0], (0, 1.2, 0.2), 0.07, S.mat(P.hexrgb("#d8d0c0"))))
        out.append(S.rod(pts[-1], (0, 1.2, 0.2), 0.07, S.mat(P.hexrgb("#d8d0c0"))))
        if name == "repeater":
            out.append(S.block((-0.8, 0.0, 0.6), (0.8, 5.0, 3.4), wood))
        return out
    if name in ("gun", "blunderbuss"):
        out = [S.block((-0.6, -4.0, -1.0), (0.6, 1.0, 0.6), wood),
               S.rod((0, 0.0, 0.0), (0, 11.0, 0.3), 0.5, steel)]
        if name == "blunderbuss":
            out.append(S.rod((0, 9.0, 0.25), (0, 12.0, 0.4), 0.55, steel, 1.2))
        return out
    raise KeyError(name)


CARRY = {
    "wood": lambda: [S.rod((-3.0, 4.2, 15.5 + k * 1.3), (3.0, 4.6, 15.3 + k * 1.3), 0.75,
                           P.bark(seed=k)) for k in range(3)],
    "gold": lambda: [S.ball((0, 4.6, 15.6), (2.6, 2.0, 2.2), cloth(P.hexrgb("#6a5a40"))),
                     S.ball((0.6, 4.4, 17.6), 0.9, P.metal(P.GOLD, shine=0.9))],
    "stone": lambda: [S.hull([(-2.4, 3.0, 14.5), (2.2, 3.4, 14.6), (0, 6.2, 14.8), (-1, 4.6, 18.0), (1.4, 4.4, 17.4)],
                             P.mottled(P.hexrgb("#7a7884"), P.hexrgb("#45434e"), scale=1.5))],
    "food": lambda: [S.Ellipsoid(P.pumpkin(), S._orient((0, 4.6, 16.0)), radii=(2.8, 2.6, 2.2))],
    "meat": lambda: [S.ball((0, 4.6, 16.0), (2.4, 2.0, 1.8), flesh(P.hexrgb("#8a2a24"), P.hexrgb("#4a1012"))),
                     S.rod((-2.6, 4.6, 16.0), (-3.8, 4.6, 16.4), 0.5, P.bone())],
    "fish": lambda: [S.ball((0, 4.4, 16.0), (3.4, 1.0, 1.4), P.mottled(P.hexrgb("#6a7a80"), P.hexrgb("#2a3a40"))),
                     S.hull([(-3.2, 4.4, 16.0), (-4.8, 4.4, 17.2), (-4.8, 4.4, 14.8)],
                            P.mottled(P.hexrgb("#4a5a60")))],
    "berries": lambda: [S.rod((0, 4.6, 14.2), (0, 4.6, 16.4), 2.4, P.wood(P.hexrgb("#8a6a3a"), P.hexrgb("#4a3418"),
                                                                        along=0, scale=0.3), 2.8)] +
                       [S.ball((x, 4.6 + y, 16.6), 0.7, S.mat(P.hexrgb("#4a1a5a"), shine=0.5))
                        for x, y in ((-1, 0), (1, 0.3), (0, -0.8), (0.2, 0.9))],
}


def _moved(shape, by):
    shape.xf = S.affine(None, tuple(np.asarray(by, float))) @ shape.xf
    return shape


def _moved_all(shapes, by):
    return [_moved(s, by) for s in shapes]


# --------------------------------------------------------------------------- the assembled person

def person(key: str, name: str, replaces: str, b: Body, g: Gear = Gear(), attack: str = "chop",
           group: str = "foot", civ: Optional[str] = None, seed: int = 0, scale: float = 1.0, rig: str = "biped"
           ) -> Unit:
    h, sh = b.hip, b.shoulder
    lx, ax = b.leg_x, b.arm_x

    def leg(nm, x):
        shapes = []
        if not b.floating:
            knee = (x, 0.3, h * 0.54)
            ankle = (x, 0.0, 1.4)
            thick = {"werewolf": 1.3, "imp": 0.95, "batman": 1.0}.get(b.kind, 0.85)
            shapes = _limb_shapes(b, (x, 0, h - 0.4), knee, thick, thick * 0.82) + \
                _limb_shapes(b, knee, ankle, thick * 0.82, thick * 0.66)
            foot_m = b.skin if b.kind not in ("knight",) else P.metal(P.hexrgb("#2a262e"), seed=seed)
            if g.armour == "plate" and b.kind != "knight":
                foot_m = P.metal(P.hexrgb("#7a7a84"), rust=0.6, seed=seed)
            shapes.append(S.ball((x, 1.0, 0.7), (1.05, 1.9, 0.8), foot_m))
            if b.kind == "knight" or g.armour == "plate":
                shapes += S.limb((x, 0.2, h * 0.54 + 1.4), (x, 0.0, 2.0), 1.05, foot_m, 0.9)
        return Part(nm, pivot=(x, 0, h), boxes=shapes)

    torso_shapes = []
    if b.kind in ("skeleton", "calavera"):
        from .undead import _ribs_material
        torso_shapes += [S.ball((0, 0, h + 0.4), (3.2, 2.0, 1.5), b.skin)]
        torso_shapes += [S.ball((0, -0.8, z), (0.75, 0.75, 0.55), b.skin) for z in np.linspace(h + 1.6, sh - 2.0, 5)]
        torso_shapes.append(S.ball((0, 0.2, sh - 3.2), (3.6, 2.7, 3.9), _ribs_material(seed)).clip((0, 0, -1), 2.6))
        torso_shapes.append(S.rod((-ax + 0.6, 0.1, sh + 0.6), (ax - 0.6, 0.1, sh + 0.6), 0.55, b.skin))
    elif b.kind == "knight":
        plate = P.metal(P.hexrgb("#2e2a32"), rust=0.3, seed=seed, shine=0.7)
        torso_shapes += [S.ball((0, 0.2, sh - 3.4), (4.1, 3.0, 4.6), plate),
                         S.rod((0, 0, h + 1.0), (0, 0, h - 2.6), 3.6, plate, 4.0)]
    else:
        chest_r = {"werewolf": (4.4, 3.2, 5.0), "imp": (3.2, 2.4, 3.6), "ghoul": (3.2, 2.4, 4.4),
                   "vampire": (3.6, 2.4, 4.4)}.get(b.kind, (3.6, 2.6, 4.4))
        torso_shapes += [S.ball((0, 0.2, sh - 3.6), chest_r, b.skin),
                         S.ball((0, 0.2, h + 1.6), (3.0, 2.2, 2.8), b.skin)]
    torso_shapes += _armour_shapes(b, g, seed) + _cloth_shapes(b, g, seed)
    if b.kind == "imp":  # a tail
        tail_m = b.skin
        pts = [np.array([0, -1.6, h + 0.8]), np.array([0.6, -4.0, h - 1.0]), np.array([1.6, -6.0, h + 1.2]),
               np.array([1.2, -7.2, h + 3.4])]
        for a, c in zip(pts, pts[1:]):
            torso_shapes += S.limb(a, c, 0.45, tail_m, 0.3)
        torso_shapes.append(S.hull([pts[-1] + (0, 0, 1.4), pts[-1] + (-0.8, 0, -0.2), pts[-1] + (0.8, 0, -0.2),
                                    pts[-1] + (0, -0.6, 0.2)], S.mat(P.hexrgb("#1a0a0a"))))
    if g.back == "wings" or b.kind in ("batman", "imp"):  # folded leathery wings on the back
        wing = S.Material(color=lambda l: P.mix(P.hexrgb("#6a3430"), P.hexrgb("#2a1214"),
                                                0.4 + 0.4 * P.noise(l.p, 1.5, seed)), vivid=0.2)
        size = 1.0 if b.kind != "imp" else 0.6
        for sx in (1, -1):
            torso_shapes.append(S.hull([(sx * 1.0, -2.4, sh + 0.5), (sx * 1.2, -2.8, sh - 1.0),
                                        (sx * 6.5 * size, -4.2, sh + 4.5 * size), (sx * 5.0 * size, -3.6, h - 2.0),
                                        (sx * 3.0 * size, -3.0, h - 3.0 * size)], wing))
    if g.back == "quiver":
        torso_shapes += [S.rod((1.6, -2.8, sh - 6.0), (-1.4, -3.2, sh + 1.0), 1.0,
                               P.wood(P.hexrgb("#5a3a22"), P.hexrgb("#2a1a10"), along=2)),
                         *[S.rod((-1.4 + k * 0.4, -3.2, sh + 0.8), (-1.6 + k * 0.4, -3.4, sh + 2.6), 0.1,
                                 S.mat(P.hexrgb("#d8d0c0"))) for k in range(3)]]
    if g.back == "keg":
        pass

    def arm(nm, x, sign, item_name, item):
        elbow = (x + sign * 0.3, 0.4, sh - 5.0)
        hand = np.array([x + sign * 0.2, 0.9, sh - 9.4])
        thick = {"werewolf": 1.15, "ghoul": 0.7, "knight": 1.05}.get(b.kind, 0.85)
        shapes = []
        if b.kind in ("skeleton", "calavera"):
            shapes.append(S.ball((x, 0, sh), 1.0, b.skin))
        shapes += _limb_shapes(b, (x, 0, sh - 0.3), elbow, thick, thick * 0.85) + \
            _limb_shapes(b, elbow, tuple(hand), thick * 0.85, thick * 0.72)
        hand_m = b.skin
        if b.kind == "knight" or g.armour == "plate":
            hand_m = P.metal(P.hexrgb("#2a262e") if b.kind == "knight" else P.hexrgb("#7a7a84"), seed=seed)
            shapes += S.limb((x, 0, sh - 0.8), elbow, thick * 1.25, hand_m, thick * 1.15)
        if g.cloth in ("robe", "wraps", "dress"):  # wide sleeves, the robe's colour
            sleeve = {"robe": cloth(P.hexrgb("#24202a"), seed=seed), "wraps": cloth(P.hexrgb("#38586a"), seed=seed),
                      "dress": ghostly(P.hexrgb("#c8d8e0"), seed + 2) if b.kind in ("ghost", "banshee")
                      else cloth(DRAB[seed % len(DRAB)], seed=seed)}[g.cloth]
            shapes.append(S.rod((x, 0, sh - 0.4), (x + sign * 0.5, 0.6, sh - 7.2), thick * 1.5, sleeve,
                                thick * 2.4))
        shapes.append(S.ball(tuple(hand), (0.8, 0.95, 1.0), hand_m))
        if b.kind in ("werewolf", "banshee", "ghoul", "batman"):  # claws
            for k in (-1, 0, 1):
                shapes.append(S.rod(hand + (k * 0.4, 0.4, -0.6), hand + (k * 0.5, 1.4, -1.8), 0.18,
                                    S.mat(P.hexrgb("#e8e0d0")), 0.02))
        part = Part(nm, pivot=(x, 0, sh), rot=(b.arms_out, 0, 0), boxes=shapes)
        if item:
            part.add(Part(item_name, pivot=tuple(hand), boxes=_moved_all(item, hand)))
        return part

    right = weapon(g.right, seed) if g.right else None
    left = weapon(g.left, seed) if g.left else None
    if g.shield:
        left = (left or []) + _shield(g.shield, seed)
    arm_r = arm("arm_r", ax, 1, "item_r", right)
    arm_l = arm("arm_l", -ax, -1, "item_l", left)
    head = Part("head", pivot=(0, 0, b.neck), boxes=[S.rod((0, -0.4, b.neck - 1.6), (0, -0.2, b.neck + 0.6), 0.8,
                                                           b.skin)] + _head_shapes(b, seed) +
                (_helmet_shapes(b, g.helmet, seed) if g.helmet else []))
    torso = Part("torso", pivot=(0, 0, h), rot=(-b.hunch, 0, 0), boxes=torso_shapes).add(head, arm_r, arm_l)
    if g.chest:
        torso.add(Part("item_c", pivot=(ax - 2.0, 3.6, sh - 3.0),
                       boxes=_moved_all(chest_weapon(g.chest, seed), (ax - 2.0, 3.6, sh - 3.0))))
    if g.carry:
        torso.add(Part("carry", boxes=CARRY[g.carry]()))
    if g.cape:
        torso.add(Part("cape", pivot=(0, -2.4, sh + 0.5), rot=(-6, 0, 0), boxes=[
            S.hull([(-3.6, -2.4, sh + 0.6), (3.6, -2.4, sh + 0.6), (-4.6, -3.4, h - 7.0), (4.6, -3.4, h - 7.0),
                    (-3.6, -3.0, sh + 0.6), (3.6, -3.0, sh + 0.6), (-4.6, -4.0, h - 7.0), (4.6, -4.0, h - 7.0)],
                   team_cloth(0.45, seed))]))
    root = Part("root").add(torso, leg("leg_r", lx), leg("leg_l", -lx))
    if b.floating:
        root.offset = (0.0, 0.0, 2.5)
    return Unit(key, name, replaces, rig, root, group=group, attack=attack, civ=civ, scale=scale,
                carry=g.carry is not None)


def _shield(kind: str, seed: int) -> list:
    """A shield on the left hand, facing out to the left (-x), painted in the owner's colour."""
    def pattern(look):
        p = look.p
        cross = (np.abs(p[:, 1]) < 0.6) | (np.abs(p[:, 2]) < 0.6)
        return np.where(cross, 0.4, 1.0)
    face = P.team(0.62, pattern)
    rimm = P.metal(rust=0.6, seed=seed)
    if kind == "round":
        return [S.rod((-0.6, 0.6, 0.6), (-1.5, 0.6, 0.6), 4.2, face),
                S.torus((-1.5, 0.6, 0.6), 4.2, 0.35, rimm, rot=(0, 90, 0)),
                S.ball((-1.6, 0.6, 0.6), (0.6, 1.2, 1.2), P.metal(rust=0.3))]
    if kind == "kite":
        pts = []
        for x in (-0.6, -1.4):
            pts += [(x, 0.6 + 3.4, 3.6), (x, 0.6 - 3.4, 3.6), (x, 0.6 + 3.0, 1.0), (x, 0.6 - 3.0, 1.0),
                    (x, 0.6, -6.0)]
        return [S.hull(pts, face)]
    if kind == "oval":
        return [S.Ellipsoid(face, S._orient((-1.2, 0.6, 0.4)), radii=(0.5, 3.4, 5.6))]
    raise KeyError(kind)
