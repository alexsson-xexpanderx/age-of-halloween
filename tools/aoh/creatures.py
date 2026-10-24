"""The creature kit: every walking person of Age of Halloween is a body, gear and what they hold.

Bodies stand on the joints Age of Minecraft's people had, so animation.py's rigs move them unchanged: "root" with
"torso" (pivot at the hips) carrying "head", "arm_r", "arm_l" (pivots at the shoulders), and "leg_r", "leg_l" (pivots
at the hips). What a hand holds is a part "item_r" / "item_l" pivoting at the hand; a two-handed crossbow or gun is
"item_c" on the torso; a load carried in both arms is "carry" (shown only while carrying); a cape is "cape".

A body kind sets the skin, head, limbs and proportions:

    skeleton   bone, ribs, a skull with glowing eyes          zombie    rotting grey-green flesh, torn clothes
    ghoul      gaunt, grey, long thin hair, hunched           imp       small, red-black, horns, wings, a tail
    werewolf   fur, a wolf's head, claws                      mummy     bandages wound round, glowing eyes
    vampire    pale, slicked hair, red eyes                   batman    a bat's head and folded wings (Camazotz)
    calavera   a painted sugar skull (Day of the Dead)        ghost     pale and glowing, floating, no legs
    jiangshi   a hopping corpse in a robe and a hat           knight    a death knight: a skull in a horned helm
    draugr     a Norse corpse, icy blue eyes                  banshee   a floating wailing woman, white hair
    phantom    a pale glowing spectre in a hooded shroud whose hem trails into mist, a dark face with two red
               glowing eyes, floating (the villagers)

Model units: about 30 high, facing +y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from . import anatomy as A
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
    return S.Material(color=color, shine=0.03, gloss=8, vivid=0.3)  # matte: shiny skin looked like plastic


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


def spectral(length: float, seed=0, top=P.hexrgb("#dcf0fc"), bottom=P.hexrgb("#3a6280")) -> S.Material:
    """A phantom's shroud: shining pale at the top, fading to a dim blue along the shape (its local z, 0 to
    `length`: rods run from their first point to their second), wisps of light through it."""
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)

    def fade(look):
        return np.clip(look.p[:, 2] / length, 0, 1) ** 0.9

    def color(look):
        f = fade(look)
        return P.mix(P.mix(top, bottom, f), bottom * 0.7, 0.35 * P.fbm(look.p, 2.2, seed))

    def glow(look):
        return 0.78 - 0.45 * fade(look) + 0.15 * (P.noise(look.p, 1.6, seed + 3) - 0.5)
    return S.Material(color=color, glow=glow, vivid=0.85)


PLATE = P.hexrgb("#5e4a3e")  # armour: dark, rusted iron (the player asked for no silver)
KNIGHT_PLATE = P.hexrgb("#2c2422")  # the death knights' blackened iron: warm, not the blue-grey of a machine
BRONZE = P.hexrgb("#7a5c36")  # helmets: tarnished bronze
BLADE = P.hexrgb("#6a5e54")  # blades: dull, rusty iron


def lamellar(seed=0) -> S.Material:
    """Rows of little plates lacquered a dark red-brown, laced together in the owner's colour (all in the owner's
    colour, the ghost samurai looked like a robot)."""
    lacquer = P.hexrgb("#4a2a22")

    def lacing(look):
        return (look.p[:, 2] * 0.4 + 0.2) % 1.0 < 0.26  # a broad band every few rows

    def color(look):
        c = P.mix(lacquer, lacquer * 0.55, P.noise(look.p, 1.5, seed))
        return np.where(lacing(look)[:, None], 0.62, c)
    return S.Material(color=color, team=lacing, shine=0.15, gloss=14)


def chainmail(base=P.hexrgb("#5e4c3c"), seed=0) -> S.Material:
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
    eye: tuple = (1.0, 0.14, 0.06)  # the eyes' glow colour
    size: float = 1.0  # the head's size
    sculpted: bool = False  # a skeleton's skull modelled bone by bone (brow, cheekbones, sockets), not one ball


SKIN = {
    "zombie": lambda s: replace(flesh(P.hexrgb("#a2b886"), P.hexrgb("#5e6e48"), s, wounds=0.6, stitches=True),
                                vivid=0.7),
    "ghoul": lambda s: replace(flesh(P.hexrgb("#b09a84"), P.hexrgb("#5a4a40"), s, wounds=0.3), vivid=0.55),
    "imp": lambda s: flesh(P.hexrgb("#8a2a22"), P.hexrgb("#3a0e10"), s),
    "werewolf": lambda s: fur(P.hexrgb("#6a5a4a"), P.hexrgb("#2a221c"), s),
    "mummy": lambda s: bandage(seed=s),
    "vampire": lambda s: replace(flesh(P.hexrgb("#ead2c2"), P.hexrgb("#9a746a"), s), shine=0.0, vivid=0.65),
    "batman": lambda s: fur(P.hexrgb("#6a564c"), P.hexrgb("#2a1e1c"), s),
    "draugr": lambda s: replace(flesh(P.hexrgb("#7a6a5e"), P.hexrgb("#2e2420"), s, wounds=0.4), vivid=0.55),
    "jiangshi": lambda s: replace(flesh(P.hexrgb("#c4b494"), P.hexrgb("#6a5c48"), s), vivid=0.55),
    "ghost": lambda s: replace(ghostly(P.hexrgb("#7eaaee"), seed=s), vivid=0.9),
    "banshee": lambda s: replace(ghostly(P.hexrgb("#8ab2f0"), seed=s), vivid=0.9),
    "phantom": lambda s: replace(ghostly(P.hexrgb("#d4ecf8"), seed=s), vivid=0.85),
}
EYES = {"skeleton": (1.0, 0.14, 0.06), "zombie": (1.0, 0.85, 0.3), "ghoul": (0.75, 0.95, 1.0),
        "imp": (1.0, 0.75, 0.15), "werewolf": (1.0, 0.7, 0.1), "mummy": (1.0, 0.14, 0.06), "vampire": (1.0, 0.15, 0.1),
        "batman": (1.0, 0.3, 0.15), "calavera": (1.0, 0.6, 0.15), "ghost": (1.0, 0.14, 0.06),
        "jiangshi": (0.9, 0.95, 0.85), "knight": (1.0, 0.12, 0.05), "draugr": (0.45, 0.8, 1.0),
        "banshee": (0.85, 0.95, 1.0), "phantom": (1.0, 0.14, 0.06)}


RED_EYES = (1.0, 0.14, 0.06)  # the player asked for no more green eyes (2026-10-24)


def body(kind: str, seed: int = 0, **kw) -> Body:
    kw.setdefault("eye", EYES.get(kind, EYES["skeleton"]))
    kw.setdefault("sculpted", kind == "skeleton")  # every skeleton gets the modelled skull, not a ball
    b = Body(kind=kind, **kw)
    if b.skin is None:
        b.skin = SKIN[kind](seed) if kind in SKIN else P.bone(seed)
    if kind in ("zombie", "ghoul") and "hunch" not in kw:
        b.hunch = 12 if kind == "zombie" else 18
    if kind in ("ghost", "banshee", "phantom"):
        b.floating = True
    if kind == "phantom" and "hunch" not in kw:
        b.hunch = 6
    return b


def _skull_shapes(b: Body, seed: int) -> list:
    """A skull: cranium with deep glowing sockets, a nose hole, a jaw with teeth."""
    from .undead import _jaw_material, _skull_material
    n = b.neck
    k = b.size
    if b.kind == "calavera":
        return _sculpted_skull(n, k, b.eye, seed) + _sugar(n, k, seed)
    if b.sculpted:
        return _sculpted_skull(n, k, b.eye, seed)
    skull_m = _skull_material(b.eye, seed)
    if b.kind == "calavera":
        skull_m = _calavera_material(b.eye, seed)
    return [S.Ellipsoid(skull_m, S._orient((0, 0.4, n + 3.6 * k)), radii=(3.4 * k, 3.6 * k, 3.3 * k)),
            S.Ellipsoid(_jaw_material(seed), S._orient((0, 1.7 * k, n + 1.0 * k)), radii=(2.3 * k, 2.1 * k,
                                                                                          1.25 * k))]


def _sculpted_skull(n: float, k: float, glow, seed: int) -> list:
    """A skull modelled bone by bone: the cranium, a narrower face under a heavy brow ridge, cheekbones, dark eye
    sockets each with a glowing point deep inside, a nose hole, and the jaw with its teeth."""
    from .undead import _jaw_material
    bone = P.bone(seed)
    c = np.array([0.0, 0.3, n + 3.8 * k])
    face_at = c + (0, 1.0 * k, -0.9 * k)
    face = eyes_material(bone, [(-0.95 * k, 0.85 * k, 0.72 * k, 0.62 * k), (0.95 * k, 0.85 * k, 0.72 * k, 0.62 * k)],
                         glow, 0.42, P.hexrgb("#120a0c"))
    nose = S.hull([face_at + (-0.38 * k, 1.78 * k, 0.05 * k), face_at + (0.38 * k, 1.78 * k, 0.05 * k),
                   face_at + (0, 1.95 * k, -0.75 * k), face_at + (0, 1.3 * k, -0.2 * k)], S.mat(P.hexrgb("#120a0c")))
    out = [S.Ellipsoid(bone, S._orient(tuple(c + (0, -0.6 * k, 0.1 * k))), radii=(2.9 * k, 3.1 * k, 2.7 * k)),
           S.Ellipsoid(face, S._orient(tuple(face_at)), radii=(2.2 * k, 1.9 * k, 2.0 * k)),
           S.Ellipsoid(bone, S._orient(tuple(c + (0, 2.0 * k, 0.6 * k))), radii=(2.3 * k, 0.8 * k, 0.6 * k)),  # the brow
           nose,
           S.Ellipsoid(_jaw_material(seed), S._orient(tuple(c + (0, 1.35 * k, -2.55 * k))),
                       radii=(1.6 * k, 1.45 * k, 0.8 * k))]
    for sx in (1, -1):  # the cheekbones
        out.append(S.Ellipsoid(bone, S._orient(tuple(c + (sx * 1.75 * k, 1.6 * k, -0.75 * k))),
                               radii=(0.7 * k, 1.0 * k, 0.5 * k)))
    return A.soft(out, "bone")


def _sugar(n: float, k: float, seed: int) -> list:
    """A sugar skull's decoration on the sculpted skull: petals round both eyes, a red heart on the nose, a row of
    coloured dots across the brow (Day of the Dead)."""
    cols = [P.hexrgb("#e0407a"), P.hexrgb("#f08a10"), P.hexrgb("#20b0b0"), P.hexrgb("#f0d020"), P.hexrgb("#8a40d0")]
    out = []
    for sx in (1, -1):
        for j in range(7):
            a = 2 * math.pi * j / 7
            p = (sx * 0.95 * k + math.cos(a) * 0.95 * k, 2.55 * k + 0.3, n + 3.75 * k + math.sin(a) * 0.85 * k)
            out.append(S.Ellipsoid(S.mat(cols[(j + seed) % len(cols)], vivid=1.0), S._orient(p),
                                   radii=(0.32 * k, 0.2 * k, 0.32 * k)))
    red = S.mat(P.hexrgb("#d01830"), vivid=1.0)
    out += [S.ball((-0.2 * k, 2.95 * k + 0.3, n + 3.05 * k), 0.24 * k, red),
            S.ball((0.2 * k, 2.95 * k + 0.3, n + 3.05 * k), 0.24 * k, red)]
    for j in range(5):
        x = (j - 2) * 0.75 * k
        out.append(S.ball((x, 2.45 * k + 0.25 - abs(x) * 0.12, n + 5.0 * k - abs(x) * 0.15), 0.22 * k,
                          S.mat(cols[(j + seed + 2) % len(cols)], vivid=1.0)))
    return out


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
    if kind == "phantom":  # no face under the hood: darkness and two glowing eyes, the hood pale and glowing
        void = S.mat(P.hexrgb("#0a0810"))
        face = eyes_material(void, [(-1.2, 0.3, 0.8, 0.5), (1.2, 0.3, 0.8, 0.5)], eye, 0.8, P.hexrgb("#0a0810"))
        hood = replace(ghostly(P.hexrgb("#d4ecf8"), seed + 4), vivid=0.85)
        top = n + 3.3 * k
        return [S.Ellipsoid(face, S._orient((0, 0.2, n + 3.0 * k)), radii=(2.6 * k, 2.8 * k, 3.2 * k)),
                S.ball((0, -0.5, top - 0.6), (3.8 * k, 4.2 * k, 4.4 * k), hood).clip((0, 1, 0), 2.4 * k),
                # the hood rises to a point at the back of the head and falls down over the shoulders
                S.hull([(-2.6, -2.0, top + 1.4), (2.6, -2.0, top + 1.4), (0, -1.0, top + 5.4), (0, -4.2, top + 1.6),
                        (-3.0, -3.4, top - 2.4), (3.0, -3.4, top - 2.4)], hood)]
    if kind == "knight":  # a skull grinning out of a dark horned helm, its sockets burning (a closed helm with two
        # eyes behind a visor looked like a robot to the player's young tester)
        return _sculpted_skull(n, k, eye, seed) + _helmet_shapes(b, "horned", seed)
    # flesh heads, modelled (the player found ball heads like Lego): a cranium, a narrower face and jaw, a brow
    # ridge over sunken sockets with glowing pupils, cheekbones, a nose, a mouth, a chin and ears
    shapes = _face_shapes(b, seed)
    c = np.array([0.0, 0.0, n + 3.4 * k])
    dark = S.mat(P.hexrgb("#16080c"))
    if kind in ("zombie", "ghoul", "draugr", "imp", "ghost"):  # a gaping mouth, a few teeth
        gape = 0.55 if kind in ("zombie", "ghoul", "ghost") else 0.35
        shapes.append(S.Ellipsoid(S.mat(P.hexrgb("#1a0a0c")), S._orient(tuple(c + (0, 2.65 * k, -1.75 * k))),
                                  radii=(0.95 * k, 0.45 * k, gape * k)))
        if kind != "ghost":
            for x in (-0.55, -0.15, 0.3, 0.6):
                shapes.append(S.block((x * k - 0.16, 2.95 * k, -1.45 * k + c[2] - 0.5), (x * k + 0.16, 3.15 * k,
                                                                                         -1.45 * k + c[2]),
                                      S.mat(P.hexrgb("#c8b890"))))
    if kind == "ghoul":  # long thin hair hanging in strings from the crown to the shoulders
        hair = S.mat(P.hexrgb("#2a2420"))
        rng = np.random.default_rng(seed + 31)
        shapes.append(S.ball(tuple(c + (0, -0.7 * k, 1.5 * k)), (2.7 * k, 2.6 * k, 1.8 * k), hair)
                      .clip((0, 1, -0.5), 0.9 * k))
        for j in range(9):
            a = math.radians(-120 + j * 30 + rng.uniform(-8, 8))
            root_ = c + (math.cos(a) * 2.5 * k, math.sin(a) * 2.2 * k - 0.6 * k, 1.4 * k)
            tip = root_ * (1.12, 1.0, 1.0) + (0, -0.8 - rng.uniform(0, 0.8), -5.0 - rng.uniform(0, 2.5))
            shapes.append(A._fine(S.rod(tuple(root_), tuple(tip), 0.42, hair, 0.12)))
    if kind == "zombie":  # matted hair in patches, one ear torn
        hair = P.mottled(P.hexrgb("#2a2018"))
        shapes += [S.ball(tuple(c + (-1.4, -0.6, 2.6)), (1.3, 1.6, 0.6), hair),
                   S.ball(tuple(c + (1.0, -1.6, 2.2)), (1.0, 1.2, 0.5), hair)]
    if kind == "imp":
        horn = S.mat(P.hexrgb("#1a1414"), shine=0.4)
        shapes += [S.rod((-1.5, 0.2, n + 5.6), (-2.8, -0.9, n + 9.0), 0.7, horn, 0.1),
                   S.rod((1.5, 0.2, n + 5.6), (2.8, -0.9, n + 9.0), 0.7, horn, 0.1)]
        for sx in (1, -1):  # long pointed ears
            shapes.append(S.hull([c + (sx * 2.6, 0.2, -0.6), c + (sx * 2.6, 0.6, 0.6), c + (sx * 2.5, -0.5, 0.0),
                                  c + (sx * 4.6, -0.6, 1.4)], b.skin))
    if kind == "draugr":  # a braided beard, frosted
        beard = P.mottled(P.hexrgb("#8a8478"), P.hexrgb("#3a3630"), 1.0, 0.5, seed)
        shapes += [S.Ellipsoid(beard, S._orient(tuple(c + (0, 2.0 * k, -2.6 * k))), radii=(1.6, 1.2, 1.4)),
                   S.rod(tuple(c + (0, 2.2 * k, -3.4 * k)), tuple(c + (0, 2.6 * k, -6.4 * k)), 0.9, beard, 0.35)]
    if kind == "vampire":  # slicked-back black hair with a widow's peak, pointed ears, sharp brows, fangs
        hair = S.mat(P.hexrgb("#141016"), shine=0.6, gloss=30)
        shapes += [S.ball(tuple(c + (0, -0.7, 1.4)), (3.0, 3.15, 2.5), hair).clip((0, 1, -0.6), 0.8),
                   S.hull([c + (-1.4, 2.0, 2.6), c + (1.4, 2.0, 2.6), c + (0, 2.55, 1.5), c + (0, 1.6, 3.0),
                           c + (0, 1.2, 2.4)], hair),  # the widow's peak
                   S.hull([c + (-0.7, 2.95, -1.65), c + (-0.4, 2.95, -1.65), c + (-0.56, 3.0, -2.35)], S.mat(P.BONE)),
                   S.hull([c + (0.7, 2.95, -1.65), c + (0.4, 2.95, -1.65), c + (0.56, 3.0, -2.35)], S.mat(P.BONE))]
        for sx in (1, -1):
            shapes += [S.hull([c + (sx * 2.6, 0.2, -0.8), c + (sx * 2.6, 0.7, 0.6), c + (sx * 2.5, -0.5, 0.0),
                               (sx * 4.0, -0.7, c[2] + 1.9)], b.skin),  # pointed ears
                       S.rod(tuple(c + (sx * 0.3, 2.85, 0.95)), tuple(c + (sx * 1.9, 2.35, 1.55)), 0.22, dark, 0.13)]
    if kind == "jiangshi":  # the Qing hat and the yellow paper talisman on the forehead
        hat = S.mat(P.hexrgb("#1a1418"))
        shapes += [S.rod((0, 0.3, n + 5.6), (0, 0.3, n + 6.6), 3.6, hat, 3.1),
                   S.ball((0, 0.3, n + 6.6), (3.1, 3.1, 1.6), hat),
                   S.ball((0, 0.3, n + 8.0), 0.6, P.team(0.6)),
                   S.block((-1.0, 3.3, n + 0.6), (1.0, 3.6, n + 5.2),
                           S.Material(color=_talisman, vivid=0.8))]
    if kind == "banshee":  # long white hair streaming down the back
        hair = ghostly(P.hexrgb("#f2f4f6"), seed + 3)
        shapes += [S.ball((0, -0.8, n + 4.0), (3.2, 3.3, 3.1), hair).clip((0, 1, 0), 0.6),
                   S.rod((0, -2.2, n + 4.0), (0, -3.4, n - 9.0), 2.8, hair, 0.8)]
    return shapes



def _face_shapes(b: Body, seed: int) -> list:
    """A modelled head of flesh (or bandage, or ghost): the cranium, a narrower face and jaw under it, a brow ridge,
    sunken dark sockets with a glowing pupil in each, cheekbones, a nose, a mouth, a chin and ears."""
    n, k = b.neck, b.size
    skin = b.skin
    dark = S.mat(P.hexrgb("#120a0c"))
    glow = S.mat(tuple(b.eye), glow=1.0)
    c = np.array([0.0, 0.0, n + 3.4 * k])
    out = [S.Ellipsoid(skin, S._orient(tuple(c + (0, -0.4 * k, 0.3 * k))), radii=(2.85 * k, 3.0 * k, 3.0 * k)),
           S.Ellipsoid(skin, S._orient(tuple(c + (0, 0.9 * k, -1.0 * k))), radii=(2.25 * k, 2.1 * k, 2.3 * k)),
           S.Ellipsoid(skin, S._orient(tuple(c + (0, 2.15 * k, 0.75 * k))), radii=(2.3 * k, 0.75 * k, 0.55 * k)),
           S.Ellipsoid(skin, S._orient(tuple(c + (0, 1.9 * k, -2.4 * k))), radii=(1.2 * k, 1.0 * k, 0.9 * k)),
           S.hull([c + (0, 2.3 * k, 0.55 * k), c + (-0.45 * k, 2.4 * k, -0.9 * k), c + (0.45 * k, 2.4 * k, -0.9 * k),
                   c + (0, 3.35 * k, -0.85 * k), c + (0, 2.0 * k, -0.3 * k)], skin),
           S.Ellipsoid(dark, S._orient(tuple(c + (0, 2.75 * k, -1.55 * k))), radii=(0.95 * k, 0.3 * k, 0.2 * k))]
    for sx in (1, -1):
        out += [S.Ellipsoid(skin, S._orient(tuple(c + (sx * 1.55 * k, 1.85 * k, -0.45 * k))),
                            radii=(0.75 * k, 0.9 * k, 0.55 * k)),
                S.Ellipsoid(dark, S._orient(tuple(c + (sx * 0.95 * k, 2.2 * k, 0.15 * k))),
                            radii=(0.62 * k, 0.4 * k, 0.42 * k)),
                S.ball(tuple(c + (sx * 0.95 * k, 2.5 * k, 0.15 * k)), 0.22 * k, glow)]
        if b.kind not in ("vampire", "imp"):
            out.append(S.Ellipsoid(skin, S._orient(tuple(c + (sx * 2.75 * k, -0.2 * k, -0.1 * k))),
                                   radii=(0.35 * k, 0.8 * k, 1.0 * k)))
    return A.soft(out, "flesh")

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
    cloth: Optional[str] = None  # loin, tunic, tabard, robe, rags, dress, wraps, shroud
    cape: bool = False
    armour: Optional[str] = None  # plate, mail, lamellar
    shield: Optional[str] = None  # round, kite, oval
    right: Optional[str] = None  # what the right hand holds (see WEAPONS)
    left: Optional[str] = None
    chest: Optional[str] = None  # crossbow, gun, blunderbuss (two-handed, held at the chest)
    carry: Optional[str] = None  # what a villager carries home (wood, gold, stone, food, meat, fish, berries)
    back: Optional[str] = None  # quiver, wings, banner, keg


def _armour_shapes(b: Body, g: Gear, seed: int) -> list:
    """Armour shaped to the body, not a barrel: a breastplate with a ridge down its front, a gorget, layered
    shoulder plates and a short skirt of plates; a chain shirt with a mail skirt; lamellar of little plates laced
    in the owner's colour."""
    sh, h = b.shoulder, b.hip
    out = []
    if g.armour == "plate":
        plate = P.metal(KNIGHT_PLATE if b.kind == "knight" else PLATE, rust=0.75, seed=seed, shine=0.15)
        out += [S.Ellipsoid(plate, S._orient((0, 0.3, sh - 3.2)), radii=(3.3, 2.15, 3.7)),
                S.Ellipsoid(plate, S._orient((0, 2.2, sh - 3.6)), radii=(0.4, 0.35, 2.9)),  # the ridge
                S.Ellipsoid(plate, S._orient((0, -0.1, sh + 0.55)), radii=(1.7, 1.45, 0.75))]  # the gorget
        out += A.drape(plate, h + 2.6, h - 1.6, 3.1, 3.6, seed, 0)  # the fauld over the hips
        for sx in (1, -1):  # pauldrons: two flat plates on each shoulder, sloping down and out over the arm
            out += [S.Ellipsoid(plate, S._orient((sx * (b.arm_x + 0.1), 0.0, sh + 0.1), (0, sx * 28, 0)),
                                radii=(2.1, 1.8, 0.7)),
                    S.Ellipsoid(plate, S._orient((sx * (b.arm_x + 0.6), 0.0, sh - 0.9), (0, sx * 40, 0)),
                                radii=(1.8, 1.6, 0.6))]
    elif g.armour == "mail":
        mail = chainmail(seed=seed)
        out += [S.Ellipsoid(mail, S._orient((0, 0.25, sh - 3.1)), radii=(3.35, 2.3, 3.5)),
                S.Ellipsoid(mail, S._orient((0, 0.1, h + 3.2)), radii=(2.95, 2.15, 2.5))]
        out += A.drape(mail, h + 1.8, h - 3.2, 3.0, 3.6, seed, 0)  # (its short sleeves are on the arms)
    elif g.armour == "lamellar":
        lam = lamellar(seed)
        out += [S.Ellipsoid(lam, S._orient((0, 0.3, sh - 3.2)), radii=(3.3, 2.25, 3.6)),
                S.Ellipsoid(lam, S._orient((0, 0.1, h + 3.2)), radii=(2.95, 2.15, 2.5))]
        out += A.drape(lam, h + 1.8, h - 2.6, 3.0, 3.5, seed, 0)
        for sx in (1, -1):  # the sode: broad curved plates hanging over the upper arms
            out.append(S.Ellipsoid(lam, S._orient((sx * (b.arm_x + 0.5), 0.0, sh - 1.6), (0, sx * 18, 0)),
                                   radii=(0.6, 1.9, 2.4)))
    return out


DRAB = [P.hexrgb("#80684e"), P.hexrgb("#7a5e3a"), P.hexrgb("#6c7250"), P.hexrgb("#74504a"), P.hexrgb("#5e4a5e")]
A_SLEEVE = {"rags": cloth(P.hexrgb("#4a4236"), seed=7)}  # the sleeves' cloth where it is not the shirt's


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
    rope = S.mat(P.hexrgb("#5a4a32"))
    leather = S.mat(P.hexrgb("#3e2a1c"), shine=0.1)
    if g.cloth == "loin":
        return A.drape(team_cloth(0.55, seed), h + 1.0, h - 3.6, 2.9, 3.4, seed, 7) + A.belt(h + 1.0, 2.9, rope, seed,
                                                                                            False)
    if g.cloth == "tunic":
        if b.kind in ("skeleton", "calavera"):  # no shirt over the ribs: a sash across them, rags round the hips
            sash = S.rod((-3.4, 2.2, sh + 0.4), (3.0, 2.4, h + 1.2), 0.7, team_cloth(0.62, seed))
            back = S.rod((-3.4, -2.4, sh + 0.4), (3.0, -2.2, h + 1.2), 0.7, team_cloth(0.5, seed))
            return [sash, back] + A.drape(drab, h + 1.2, h - 3.4, 2.7, 3.4, seed, 9) + A.belt(h + 1.2, 2.8, rope,
                                                                                             seed, False)
        return (A.torso(b.kind, drab, h + 0.05, sh + 0.05, b.arm_x + 0.1)[:2] +
                A.drape(drab, h + 1.4, h - 3.6, 3.0, 3.7, seed, 10) + A.belt(h + 1.4, 3.0, team_cloth(0.62, seed),
                                                                              seed))
    if g.cloth == "tabard":  # narrow panels front and back over the chest, in the owner's colour, a skull on it
        def emblem(look):
            p = look.p
            skull = (np.hypot(p[:, 0] / 1.0, (p[:, 2] - (sh - 4.5)) / 1.1) < 1.0)
            return np.where(skull, 0.25, 1.0)
        t = P.team(0.6, emblem)
        out = A.drape(t, sh - 0.4, h + 1.2, 3.25, 3.15, seed, 0, y_scale=0.86) + \
            A.drape(t, h + 1.3, h - 4.2, 3.05, 3.5, seed + 1, 5, y_scale=0.9)
        for s in out:  # open at the sides: a panel in front, one behind
            s.clip((1, 0, 0), 2.3).clip((-1, 0, 0), 2.3)
        return out + A.belt(h + 1.2, 2.9, S.mat(P.hexrgb("#3e2a1c"), shine=0.1), seed, False)
    if g.cloth == "robe":  # a long dark robe falling in folds to a torn hem, a sash in the owner's colour
        dark = cloth(P.hexrgb("#24202a"), seed=seed)
        return (A.drape(dark, sh + 0.3, h - 1.0, 3.1, 4.0, seed, 0) +
                A.drape(dark, h - 0.9, 0.6, 4.0, 4.9, seed + 1, 12) + A.belt(h + 1.4, 3.0, team_cloth(0.55, seed), seed))
    if g.cloth == "rags":  # a torn shirt, ragged breeches hanging in strips, a belt in the owner's colour
        shirt = A.torso(b.kind, drab, h + 0.05, sh + 0.05, b.arm_x + 0.1)[:2]
        return shirt + A.drape(cloth(P.hexrgb("#3e382e"), seed=seed), h + 1.0, h - 4.4, 3.0, 3.6, seed, 11) + \
            A.belt(h + 1.0, 3.0, team_cloth(0.6, seed), seed)
    if g.cloth == "shroud":  # a long hooded robe, pale and glowing, its hem trailing into wisps of mist
        robe_top, robe_foot = sh + 0.6, 1.6
        rng = np.random.default_rng(seed + 40)
        out = [S.rod((0, 0, robe_top), (0, -0.4, robe_foot), 2.9, spectral(robe_top - robe_foot, seed + 2), 5.0),
               S.rod((0, -0.3, sh + 2.0), (0, -0.3, sh - 2.2), 2.2, spectral(14.0, seed + 5), 5.0),  # the cape
               _belt(b, h + 1.6, True, seed),
               S.rod((-3.0, 2.6, sh + 0.2), (2.6, 3.2, h + 1.6), 0.55, team_cloth(0.6, seed))]  # a sash
        for j in range(11):  # the tatters: tongues of the hem drifting back and down into mist
            a = 2 * math.pi * j / 11 + rng.uniform(-0.2, 0.2)
            top = np.array([math.cos(a) * 4.6, math.sin(a) * 4.6 - 0.4, 3.0])
            drift = (rng.uniform(-0.8, 0.8), -2.2 - rng.uniform(0, 2.0), -1.8 - rng.uniform(0, 1.4))
            tip = top * (1.15, 1.0, 0) + drift
            out.append(S.rod(top, tip, 1.3, spectral(6.0, seed + 6 + j, P.hexrgb("#8ab0c8")), 0.05))
        for j in range(4):  # mist round the foot, glowing faintly, casting nothing
            m = S.Ellipsoid(ghostly(P.hexrgb("#7ea2b8"), seed + 9 + j),
                            S._orient((rng.uniform(-2.5, 2.5), rng.uniform(-3.5, 1.5), -1.2)),
                            radii=(rng.uniform(2.4, 3.6), rng.uniform(1.8, 2.8), 0.7))
            m.casts = False
            m.ink = False
            out.append(m)
        return out
    if g.cloth == "dress":
        gown = replace(ghostly(P.hexrgb("#7aa6ea"), seed + 2), vivid=0.9) if b.kind in ("ghost", "banshee") else drab
        return A.drape(gown, sh + 0.2, h - 2.0, 3.0, 4.6, seed, 0) + A.drape(gown, h - 1.9, -1.0, 4.6, 6.4, seed + 1,
                                                                                     13) + \
            A.belt(h + 2.0, 2.9, team_cloth(0.6, seed), seed, False)
    if g.cloth == "hakama":  # wide pleated trousers flaring from under the armour's skirt, a ghost's trailing away
        gown = replace(ghostly(P.hexrgb("#6a8ed8"), seed + 2), vivid=0.9) if b.floating else             cloth(P.hexrgb("#2a2430"), seed=seed)
        return A.drape(gown, h + 0.5, -1.0, 2.9, 5.2, seed + 1, 13)
    if g.cloth == "wraps":  # a long Qing robe, dark, a gold hem, the owner's colour in the sash
        dark = cloth(P.hexrgb("#38586a"), seed=seed)
        return (A.drape(dark, sh + 0.5, h - 1.0, 3.3, 3.9, seed, 0) + A.drape(dark, h - 0.9, 0.8, 3.9, 4.6, seed + 1, 0)
                + A.belt(h + 1.2, 3.1, team_cloth(0.6, seed), seed, False))
    return []


def _helmet_shapes(b: Body, kind: str, seed: int) -> list:
    """Hats and helmets, fitted to the modelled heads: the cranium is centred about 3.8 above the neck (times the
    head's size) and a little back, reaches 6.7 up, the brow sits at 4.2 and the eyes at 3.6. A helmet comes down to
    the brow and no lower (lower, it hid the eyes, or the whole face under the ghost pirate's hat); the top of the
    skull must not poke through."""
    n, k = b.neck, b.size
    top = n + 3.6 * k
    zc, yc = n + 3.8 * k, -0.35 * k  # the cranium's middle
    iron = P.metal(BRONZE, rust=0.45, seed=seed + 3, shine=0.2)
    dark_iron = P.metal(KNIGHT_PLATE, rust=0.5, seed=seed + 3, shine=0.15)

    def open_helm(m):  # a deep rounded helm, open at the face, its sides down over the cheeks, a nasal
        out = [S.ball((0, yc, zc + 0.2 * k), (3.25 * k, 3.5 * k, 3.4 * k), m).clip((0, 1, 0.35), 1.6 * k),
               S.block((-0.3 * k, 2.95 * k, n + 2.9 * k), (0.3 * k, 3.4 * k, n + 4.7 * k), m)]
        for sx in (1, -1):
            out.append(S.Ellipsoid(m, S._orient((sx * 2.4 * k, 1.0 * k, n + 2.2 * k)), radii=(0.55, 1.3, 1.5)))
        return out
    if kind == "kettle":  # a round crown and a brim turned down, above the brow
        return [S.ball((0, yc, zc + 0.6 * k), (3.15 * k, 3.4 * k, 2.75 * k), iron).clip((0, 0, -1), -0.05 * k),
                S.rod((0, yc, n + 4.75 * k), (0, yc, n + 4.25 * k), 3.3 * k, iron, 4.1 * k)]  # the brim
    if kind == "nasal":
        return [S.ball((0, yc, zc + 0.4 * k), (3.2 * k, 3.45 * k, 3.0 * k), iron).clip((0, 0, -1), 0.25 * k),
                S.block((-0.3 * k, 2.95 * k, n + 2.9 * k), (0.3 * k, 3.4 * k, n + 4.6 * k), iron)]
    if kind == "great":  # open at the face, cheek guards, a plume in the owner's colour
        out = open_helm(iron)
        for j in range(5):
            out.append(S.Ellipsoid(P.team(0.6), S._orient((0, -0.6 - j * 0.9, n + 7.6 * k - j * 0.5),
                                                           (60 + j * 10, 0, 0)), radii=(0.7, 0.6, 1.6)))
        return out
    if kind == "horned":  # dark, the skull grinning out of it, great curling horns
        horn = S.mat(P.hexrgb("#d8ccb0"), shine=0.2)
        out = open_helm(dark_iron)
        for sx in (1, -1):
            pts = [np.array([sx * 2.9, -0.2, n + 5.2]), np.array([sx * 4.9, -0.3, n + 6.2]),
                   np.array([sx * 6.0, 0.3, n + 8.3]), np.array([sx * 5.2, 1.1, n + 9.9])]
            for j, (p, q) in enumerate(zip(pts, pts[1:])):
                out.append(S.rod(tuple(p), tuple(q), 0.95 - 0.28 * j, horn, 0.7 - 0.25 * j))
        return out
    if kind in ("hood", "cowl"):  # a hood over the head, open at the face, and a short cape over the shoulders
        hood = team_cloth(0.42, seed) if kind == "hood" else cloth(P.hexrgb("#1c181e"), seed=seed)
        mantle = cloth(DRAB[seed % len(DRAB)], seed=seed) if kind == "hood" else hood
        return [S.ball((0, yc - 0.1, zc - 0.3 * k), (3.45 * k, 3.75 * k, 3.75 * k), hood).clip((0, 1, 0), 2.0 * k),
                S.rod((0, -0.2, b.shoulder + 1.6), (0, -0.2, b.shoulder - 1.4), 2.6, mantle, 5.0)]
    if kind == "crown":  # on the top of the head, above the brow (lower, its band crossed the eyes like make-up)
        gold = P.metal(P.GOLD, seed=seed, shine=0.8)
        z0 = n + 5.6 * b.size
        out = [S.rod((0, 0.3, z0), (0, 0.3, z0 + 1.1), 2.75, gold, 2.85, capped=False)]
        for a in range(0, 360, 60):
            r = math.radians(a)
            out.append(S.hull([(math.cos(r) * 2.6 - 0.5, math.sin(r) * 2.6 + 0.3, z0 + 1.0),
                               (math.cos(r) * 2.6 + 0.5, math.sin(r) * 2.6 + 0.3, z0 + 1.0),
                               (math.cos(r) * 2.9, math.sin(r) * 2.9 + 0.3, z0 + 2.6)], gold))
        out.append(S.ball((0, 3.1, z0 + 0.55), 0.3, S.mat(P.BLOOD, glow=0.5)))
        return out
    if kind == "turban":  # wound round the head above the brow
        return [S.torus((0, yc, n + 5.0 * k), 2.75 * k, 1.25 * k, team_cloth(0.65, seed)),
                S.ball((0, yc, n + 6.1 * k), (2.95 * k, 3.1 * k, 2.2 * k), team_cloth(0.6, seed))]
    if kind == "fez":  # on the crown of the head, a gold tassel hanging from its top
        return [S.rod((0, yc, n + 5.3 * k), (0, yc, n + 8.6 * k), 2.45 * k, P.team(0.55), 2.0 * k),
                S.rod((0, yc, n + 8.6 * k), (1.6, yc - 0.9, n + 6.4 * k), 0.2, S.mat(P.GOLD))]
    if kind == "feathers":  # marigolds and plumes (Day of the Dead)
        out = []
        cols = [P.hexrgb("#f08a10"), P.hexrgb("#e0407a"), P.hexrgb("#f0c020")]
        for j, a in enumerate(range(-70, 71, 20)):
            r = math.radians(a)
            base = (math.sin(r) * 2.4, math.cos(r) * 0.6 - 1.0, n + 5.8 * k)
            tip = (math.sin(r) * 6.0, -2.0, top + 8.5 - abs(a) * 0.04)
            out.append(S.Ellipsoid(P.team(0.6) if j % 3 == 1 else S.mat(cols[j % 3], vivid=1.0),
                                   S.axis_frame(base, tip) @ S.affine(None, (0, 0, 3.6)),
                                   radii=(0.9, 0.35, 3.8)))
        out.append(S.torus((0, yc, n + 5.0 * k), 2.95 * k, 0.75 * k, S.mat(P.hexrgb("#f08a10"), vivid=1.0)))
        return out
    if kind == "pelt":  # a jaguar's head over the warrior's
        spots = S.Material(color=lambda l: np.where(
            (P.noise(l.p, 1.0, 5) > 0.62)[:, None], P.hexrgb("#1a1410"), P.hexrgb("#c08a30")), vivid=0.6)
        return [S.ball((0, -0.3, top + 0.2), (3.9, 4.3, 3.4), spots).clip((0, 1, 0.3), 1.6),
                S.ball((0, 2.6, top + 1.4), (2.6, 2.2, 1.6), spots),
                S.hull([(-1.0, 3.8, top + 0.4), (-0.6, 3.8, top + 0.4), (-0.8, 4.0, top - 0.8)], S.mat(P.BONE)),
                S.hull([(1.0, 3.8, top + 0.4), (0.6, 3.8, top + 0.4), (0.8, 4.0, top - 0.8)], S.mat(P.BONE))]
    if kind == "oni":  # a lacquered kabuto, its neck guard flaring behind, a red demon mask over mouth and chin
        red = S.mat(P.hexrgb("#8a1a14"), shine=0.2, gloss=16, vivid=0.7)
        lac = P.metal(P.hexrgb("#2e1e1a"), rust=0.3, seed=seed, shine=0.2)
        horn = P.metal(P.hexrgb("#8a6a3a"), rust=0.4, seed=seed, shine=0.25)
        out = [S.ball((0, yc, zc + 0.5 * k), (3.35 * k, 3.6 * k, 3.0 * k), lac).clip((0, 0, -1), 0.15 * k),
               S.ball((0, yc - 0.3, n + 3.9 * k), (4.3 * k, 4.4 * k, 2.6 * k), lamellar(seed))  # the neck guard
               .clip((0, 1, 0), 1.0 * k).clip((0, 0, 1), 0.9 * k),
               S.Ellipsoid(red, S._orient((0, 2.5 * k, n + 2.0 * k)), radii=(1.9 * k, 0.95 * k, 1.25 * k))]
        for sx in (1, -1):
            out += [S.hull([(sx * 0.9, 3.25 * k, n + 1.7 * k), (sx * 0.45, 3.25 * k, n + 1.7 * k),
                            (sx * 0.68, 3.4 * k, n + 0.9 * k)], S.mat(P.BONE)),  # fangs
                    S.rod((sx * 1.4, 1.4, n + 6.6 * k), (sx * 2.7, 2.0, n + 9.4 * k), 0.55, horn, 0.12)]
        return out
    if kind == "viking":  # a rounded cap, spectacles round the eyes and a guard down the nose: the face shows
        out = [S.ball((0, yc, zc + 0.6 * k), (3.2 * k, 3.45 * k, 3.0 * k), iron).clip((0, 0, -1), 0.3 * k),
               S.block((-0.28 * k, 3.4 * k, n + 2.9 * k), (0.28 * k, 3.75 * k, n + 4.6 * k), iron)]
        for sx in (1, -1):
            out.append(S.rod((sx * 0.35, 3.15 * k, n + 3.0 * k), (sx * 2.2, 2.4 * k, n + 3.25 * k), 0.26, iron))
        return out
    if kind == "tricorn":  # on top of the head, the brim above the brow
        hat = S.mat(P.hexrgb("#1a1618"))
        return [S.rod((0, yc, n + 4.9 * k), (0, yc, n + 5.4 * k), 4.7 * k, hat),
                S.ball((0, yc, n + 5.6 * k), (3.0 * k, 3.15 * k, 2.1 * k), hat).clip((0, 0, -1), 0.4 * k),
                S.ball((0, yc, n + 5.45 * k), (4.5 * k, 4.5 * k, 0.6), P.team(0.5)).clip((0, 0, -1), -0.2)]
    return []


# --------------------------------------------------------------------------- what they hold

def _forward(up_deg=20.0):
    a = math.radians(up_deg)
    return np.array([0.0, math.cos(a), math.sin(a)]), np.array([0.0, -math.sin(a), math.cos(a)])


def _blade(length=11.0, width=0.85, m=None, up_deg=20.0, start=1.6) -> list:
    d, up = _forward(up_deg)
    m = m or P.metal(BLADE, rust=0.7)
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
    steel = P.metal(BLADE, rust=0.7, seed=seed)
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
        return _blade(14.0, 0.55, P.metal(P.hexrgb("#4e4a50"), rust=0.3, seed=seed, shine=0.8), up_deg=28) + \
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
        shapes += [_moved(s, top - (0, 0, 0.6)) for s in skull((0, 0, 0), 1.5, seed, RED_EYES)]
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
    steel = P.metal(P.hexrgb("#4e4238"), rust=0.6, seed=seed)
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

    bony = b.kind in ("skeleton", "calavera")
    beast = b.kind in ("werewolf", "imp", "batman", "ghoul")
    clad = g.cloth in ("tunic", "rags") and not bony and not beast and b.kind != "mummy"

    def leg(nm, x):
        shapes = []
        if not b.floating:
            hipj, knee, ankle = (x, 0.0, h - 0.4), (x, 0.4, h * 0.52), (x, 0.0, 1.3)
            if bony:
                shapes = A.bone(hipj, knee, 0.42, b.skin) + A.bone(knee, ankle, 0.36, b.skin) + \
                    A.foot(x, b.skin, bony=True)
            elif b.kind == "knight" or g.armour == "plate":
                metal = P.metal(KNIGHT_PLATE if b.kind == "knight" else PLATE, rust=0.6, seed=seed, shine=0.15)
                shapes = A.muscle(hipj, knee, 1.3, 0.9, metal, 0.4, 1.12) + \
                    A.muscle(knee, ankle, 0.95, 0.65, metal, 0.32, 1.15, back=0.3) + \
                    [S.Ellipsoid(metal, S._orient((x, 0.75, h * 0.52)), radii=(0.95, 0.6, 0.9))] + \
                    A.foot(x, metal, "boot")
            else:
                thick = {"werewolf": 1.35, "imp": 0.95, "batman": 1.05, "ghoul": 0.85}.get(b.kind, 1.0)
                shapes = A.muscle(hipj, knee, 1.25 * thick, 0.78 * thick, b.skin, 0.35, 1.15) + \
                    A.muscle(knee, ankle, 0.8 * thick, 0.48 * thick, b.skin, 0.3, 1.2, back=0.35)
                if clad:  # loose breeches to the shin, and boots
                    pants = cloth(P.hexrgb("#3e3830") if g.cloth == "rags" else DRAB[(seed + 1) % len(DRAB)] * 0.8,
                                  seed=seed + 3)
                    shapes += A.trouser(hipj, knee, 1.5 * thick, 1.15 * thick, pants) + \
                        A.trouser(knee, (x, 0.05, 3.6), 1.15 * thick, 1.0 * thick, pants)
                    shapes += A.foot(x, S.mat(P.hexrgb("#3a2818"), shine=0.1), "boot")
                else:
                    shapes += A.foot(x, b.skin, "bare", claws=beast)
        return Part(nm, pivot=(x, 0, h), boxes=shapes)

    torso_shapes = []
    if bony:
        from .undead import _ribs_material
        torso_shapes += A.skeleton_torso(b.skin, _ribs_material(seed), h, sh, ax)
    elif b.kind == "knight":  # blackened plate under a torn surcoat in the owner's colour, a ragged black mantle
        plate = P.metal(KNIGHT_PLATE, rust=0.5, seed=seed, shine=0.15)
        torso_shapes += A.torso("knight", plate, h, sh, ax)
        torso_shapes += A.drape(team_cloth(0.45, seed), h + 2.0, h - 6.0, 3.0, 3.8, seed, 11) + \
            A.belt(h + 2.0, 3.0, S.mat(P.hexrgb("#2a1a12")), seed, False)
        torso_shapes += A.drape(cloth(P.hexrgb("#1e1a1c"), seed=seed + 4), sh + 1.2, sh - 3.4, 2.4, 4.9, seed + 4,
                                12)
    else:
        torso_shapes += A.torso(b.kind, b.skin, h, sh, ax)
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
        size = 1.0 if b.kind != "imp" else 0.85
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
    if b.kind == "vampire":  # the tall stand-up collar behind the head: black outside, blood red inside
        def lining(look):
            return np.where((look.n[:, 1] > 0.25)[:, None], P.hexrgb("#8a0c16"), P.hexrgb("#121014"))
        collar = S.Material(color=lining, shine=0.3, vivid=0.7)
        for sx in (1, -1):
            base = [(sx * 0.6, -1.9, sh + 0.6), (sx * 3.8, -1.2, sh + 0.4)]
            tip = [(sx * 1.3, -2.6, b.neck + 6.6), (sx * 5.0, -0.6, b.neck + 5.4)]
            pts = base + tip
            torso_shapes.append(S.hull(pts + [(x, y - 0.45, z) for x, y, z in pts], collar))
        torso_shapes += [S.Ellipsoid(cloth(P.hexrgb("#e8e2d8"), seed=seed), S._orient((0, 2.4, sh - 1.4)),
                                     radii=(1.3, 0.8, 1.9)),  # a white cravat
                         S.ball((0, 3.1, sh - 1.0), 0.45, S.mat(P.BLOOD, shine=0.8, glow=0.3))]  # a red brooch

    def arm(nm, x, sign, item_name, item):
        elbow = (x + sign * 0.3, 0.45, sh - 5.0)
        hand = np.array([x + sign * 0.2, 0.9, sh - 9.4])
        wrist = hand + (0, -0.15, 0.75)
        thick = {"werewolf": 1.2, "ghoul": 0.78, "knight": 1.1, "imp": 0.9}.get(b.kind, 1.0)
        shoulder = (x, 0, sh - 0.3)
        shapes = []
        hand_m = b.skin
        glove = None
        if bony:
            shapes += A.bone(shoulder, elbow, 0.38, b.skin) + A.bone(elbow, tuple(wrist), 0.32, b.skin)
        elif b.kind == "knight" or g.armour == "plate":
            glove = P.metal(KNIGHT_PLATE if b.kind == "knight" else PLATE, rust=0.6, seed=seed, shine=0.15)
            shapes += A.muscle(shoulder, elbow, 1.05 * thick, 0.8 * thick, glove, 0.4, 1.15) + \
                A.muscle(elbow, tuple(wrist), 0.8 * thick, 0.55 * thick, glove, 0.3, 1.12) + \
                [S.Ellipsoid(glove, S._orient(tuple(np.asarray(elbow) + (0, -0.35, 0))), radii=(0.9, 0.8, 0.85))]
        else:
            shapes += A.deltoid(shoulder, sign, 0.95 * thick, b.skin) + \
                A.muscle(shoulder, elbow, 0.95 * thick, 0.68 * thick, b.skin, 0.42, 1.16) + \
                A.muscle(elbow, tuple(wrist), 0.7 * thick, 0.45 * thick, b.skin, 0.3, 1.14)
            if clad:  # a loose sleeve over the upper arm
                shapes += A.sleeve(shoulder, elbow, 1.15 * thick, A_SLEEVE.get(g.cloth, cloth(DRAB[seed % len(DRAB)],
                                                                                             seed=seed)))
        if g.armour == "mail":  # the mail shirt's short sleeve
            short = tuple(np.asarray(shoulder) + (np.asarray(elbow) - shoulder) * 0.55)
            shapes += A.sleeve(shoulder, short, (0.95 if bony else 1.2) * thick, chainmail(seed=seed), 1.3)
        if g.cloth in ("robe", "wraps", "dress", "shroud"):  # wide sleeves, the robe's colour
            sleeve = {"robe": cloth(P.hexrgb("#24202a"), seed=seed), "wraps": cloth(P.hexrgb("#38586a"), seed=seed),
                      "dress": replace(ghostly(P.hexrgb("#7aa6ea"), seed + 2), vivid=0.9) if b.kind in ("ghost", "banshee")
                      else cloth(DRAB[seed % len(DRAB)], seed=seed),
                      "shroud": spectral(10.0, seed + 2)}[g.cloth]
            shapes.append(S.rod((x, 0, sh - 0.4), (x + sign * 0.5, 0.6, sh - 7.2), thick * 1.5, sleeve,
                                thick * 2.4))
        shapes += A.hand(hand, sign, hand_m, claws=b.kind in ("werewolf", "banshee", "ghoul", "batman", "imp"),
                         bony=bony, glove=glove)
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
    todo = [root]
    while todo:  # whatever else is made of the skin (heads, ears, tails, the neck) melts into the body too
        part = todo.pop()
        todo += part.children
        A.soft([s for s in part.boxes if isinstance(s, S.Shape) and s.material is b.skin], "flesh")
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
