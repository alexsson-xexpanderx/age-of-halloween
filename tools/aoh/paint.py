"""Procedural surfaces for the smooth models (sculpt.py): noise, wood, stone, shingles, pumpkin ribs, bone and bark,
and the Halloween colour set.

Every pattern is a function of the hit point in the shape's own coordinates (model units, about 45 to a tile), so
it stays put on the shape whatever the pose, and is crisp at any size. Patterns are seeded with fixed numbers, never
Python's hash(), so every build draws the same pictures.
"""
from __future__ import annotations

import numpy as np

from .sculpt import Look, Material

# --------------------------------------------------------------------------- colours

def hexrgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


PUMPKIN = hexrgb("#d8701a")
PUMPKIN_DARK = hexrgb("#7e340c")
PUMPKIN_ROT = hexrgb("#6e5a1c")
STEM = hexrgb("#5a6b2a")
CANDLE = hexrgb("#ffc23a")  # the glow inside a jack-o'-lantern
EMBER = hexrgb("#ff7a1a")
BONE = hexrgb("#e2cf9e")
BONE_DARK = hexrgb("#86704a")
GHOST = hexrgb("#d8f0ec")
SLIME = hexrgb("#7cd23a")
WITCH_GREEN = hexrgb("#6fae4a")
NIGHT = hexrgb("#2a2238")
PURPLE = hexrgb("#5b3a7a")
PURPLE_DARK = hexrgb("#352247")
MOSS = hexrgb("#5d6e3a")
RUST = hexrgb("#8a4a2a")
IRON = hexrgb("#3a3a42")
GOLD = hexrgb("#e2b23a")
WOOD = hexrgb("#6b4a32")
WOOD_DARK = hexrgb("#3e2a1e")
STONE = hexrgb("#8c8a94")
STONE_DARK = hexrgb("#5c5a66")
BLOOD = hexrgb("#8a1a1a")


# --------------------------------------------------------------------------- noise

def _hash3(i, j, k, seed):
    h = (i * 374761393 + j * 668265263 + k * 2147483647 + seed * 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def noise(p: np.ndarray, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Smooth value noise in 0..1 at points p (n, 3), one cell every `scale` units."""
    q = p / scale
    f = np.floor(q)
    t = q - f
    t = t * t * (3 - 2 * t)
    i, j, k = (f[:, 0].astype(np.int64), f[:, 1].astype(np.int64), f[:, 2].astype(np.int64))
    out = 0.0
    for dx in (0, 1):
        wx = t[:, 0] if dx else 1 - t[:, 0]
        for dy in (0, 1):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            for dz in (0, 1):
                wz = t[:, 2] if dz else 1 - t[:, 2]
                out = out + wx * wy * wz * _hash3(i + dx, j + dy, k + dz, seed)
    return out


def fbm(p, scale=1.0, seed=0, octaves=3):
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        total = total + amp * noise(p, scale / 2 ** o, seed + 17 * o)
        norm += amp
        amp *= 0.5
    return total / norm


def mix(a, b, f):
    f = np.asarray(f, float)[..., None] if np.ndim(f) else f
    return np.asarray(a, float) * (1 - f) + np.asarray(b, float) * f


# --------------------------------------------------------------------------- surfaces

def mottled(base, dark=None, scale=3.0, amount=0.18, seed=0) -> Material:
    """A plain colour, gently blotchy."""
    base = np.asarray(base, float)
    dark = base * 0.7 if dark is None else np.asarray(dark, float)

    def color(l: Look):
        return mix(base, dark, amount * 2 * (fbm(l.p, scale, seed) - 0.25))
    return Material(color=color)


def pumpkin(ribs: int = 10, base=PUMPKIN, dark=PUMPKIN_DARK, rot: float = 0.0, seed: int = 0,
            carve=None, axis: int = 2) -> Material:
    """A pumpkin's skin: ribs round the `axis`, darker in the grooves, with speckles; `rot` adds rotten patches.
    `carve(look) -> 0..1` marks carved openings, which glow with candle light."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def angle(p):
        a, b = [(0, 1), (1, 2), (0, 2)][[2, 0, 1].index(axis)]
        return np.arctan2(p[:, b], p[:, a])

    def groove(p):
        return np.abs(np.cos(angle(p) * ribs / 2)) ** 0.6  # 1 mid-rib, 0 in a groove

    def color(l: Look):
        g = groove(l.p)
        c = mix(dark, base, 0.35 + 0.65 * g)
        c = mix(c, c * 1.12, noise(l.p, 1.2, seed))
        if rot:
            r = np.clip((fbm(l.p, 6.0, seed + 5) - 0.62) * 4 * rot, 0, 1)
            c = mix(c, PUMPKIN_ROT, r)
        if carve is not None:
            k = carve(l)
            c = mix(c, CANDLE, k)
        return c

    def glow(l: Look):
        return carve(l) if carve is not None else np.zeros(len(l.p))

    def bump(l: Look):
        n = l.n.copy()
        a = angle(l.p)
        d = -np.sin(a * ribs) * 0.35  # tilt the normal round the axis across each rib
        ax_a, ax_b = [(0, 1), (1, 2), (0, 2)][[2, 0, 1].index(axis)]
        tangent = np.zeros_like(n)
        tangent[:, ax_a] = -np.sin(a)
        tangent[:, ax_b] = np.cos(a)
        n = n + tangent * d[:, None]
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    return Material(color=color, glow=glow, bump=bump, shine=0.25, gloss=18, vivid=0.7)


def wood(base=WOOD, dark=WOOD_DARK, along: int = 2, scale: float = 0.7, seed: int = 0, planks: float = 0.0
         ) -> Material:
    """Wood grain running along an axis; `planks` > 0 adds boards that wide, with dark gaps."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)
    across = [a for a in (0, 1, 2) if a != along]

    def color(l: Look):
        p = l.p
        warp = fbm(p, 4.0, seed) * 3
        rings = np.sin((p[:, across[0]] + warp) / scale * 2.2) * 0.5 + 0.5
        c = mix(base, dark, 0.15 + 0.45 * rings ** 3)
        c = mix(c, dark, 0.25 * noise(p * np.array([1, 1, 1]) / 1.0, 0.6, seed + 3))
        if planks:
            u = p[:, across[0]] / planks
            gap = np.abs(u - np.round(u)) * planks < 0.18
            board = np.floor(u)
            tint = 0.85 + 0.3 * _hash3(board.astype(np.int64), 1, 2, seed)
            c = c * tint[:, None]
            c = np.where(gap[:, None], dark * 0.6, c)
        return c
    return Material(color=color)


def bricks(base=STONE, dark=STONE_DARK, w: float = 4.0, h: float = 2.0, mortar=None, moss: float = 0.0,
           seed: int = 0, axes=(0, 2), rough=0.18) -> Material:
    """Stone blocks in courses (u along `axes[0]`, v up `axes[1]`), each block its own shade, with mortar lines;
    `moss` creeps up from the bottom and into the joints."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)
    mortar = base * 0.55 if mortar is None else np.asarray(mortar, float)

    def color(l: Look):
        p = l.p
        n = np.abs(l.n)
        # on faces looking along the u axis, run the courses along the other horizontal axis instead
        side = n[:, axes[0]] > 0.7
        other = [a for a in (0, 1) if a != axes[0]][0] if axes[0] in (0, 1) else 1
        u = np.where(side, p[:, other], p[:, axes[0]])
        v = p[:, axes[1]]
        row = np.floor(v / h)
        uu = u / w + 0.5 * (row % 2)
        col = np.floor(uu)
        fu, fv = uu - col, v / h - row
        joint = (np.minimum(fu, 1 - fu) * w < 0.22) | (np.minimum(fv, 1 - fv) * h < 0.2)
        shade = _hash3(col.astype(np.int64), row.astype(np.int64), 7, seed)
        c = mix(base, dark, 0.15 + 0.55 * shade)
        c = mix(c, dark, rough * fbm(p, 1.5, seed + 1))
        c = np.where(joint[:, None], mortar, c)
        if moss:
            m = np.clip((fbm(p, 3.0, seed + 9) - 0.45) * 3 * moss - (p[:, 2] - 2) * 0.05, 0, 1)
            m = np.where(joint, np.minimum(1, m * 1.6), m)
            c = mix(c, MOSS, m)
        return c
    return Material(color=color)


def shingles(base=PURPLE, dark=PURPLE_DARK, w: float = 2.2, h: float = 1.6, seed: int = 0, axes=(0, 1)
             ) -> Material:
    """Roof shingles: staggered rows of rounded tabs, each tab darker at its lower edge."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(l: Look):
        p = l.p
        u, v = p[:, axes[0]], p[:, axes[1]]
        row = np.floor(v / h)
        uu = u / w + 0.5 * (row % 2)
        col = np.floor(uu)
        fu, fv = uu - col, v / h - row
        round_edge = fv < 0.25 + 0.25 * (1 - (2 * fu - 1) ** 2)
        tint = _hash3(col.astype(np.int64), row.astype(np.int64), 3, seed)
        c = mix(base, dark, 0.1 + 0.35 * tint)
        c = mix(c, dark * 0.6, (1 - fv) ** 4 * 0.7)
        c = np.where((round_edge & (fv < 0.12))[:, None], dark * 0.5, c)
        return mix(c, dark, 0.2 * fbm(p, 2.0, seed + 4))
    return Material(color=color)


def bone(seed: int = 0) -> Material:
    def color(l: Look):
        stain = np.clip((fbm(l.p, 2.5, seed + 3) - 0.5) * 2.5, 0, 1)  # old, dirty bone
        return mix(mix(BONE, BONE_DARK, 0.15 + 0.5 * fbm(l.p, 1.2, seed) ** 2), hexrgb("#3e3428"), stain * 0.6)
    return Material(color=color, shine=0.2, gloss=12, vivid=0.5)


def bark(base=hexrgb("#4a3a40"), dark=hexrgb("#221a20"), seed: int = 0) -> Material:
    """Dead bark: deep vertical cracks."""
    base, dark = np.asarray(base, float), np.asarray(dark, float)

    def color(l: Look):
        p = l.p
        a = np.arctan2(p[:, 1], p[:, 0])
        crack = np.abs(np.sin(a * 5 + fbm(p, 3.0, seed) * 6 + p[:, 2] * 0.15))
        return mix(dark, base, np.clip(crack * 1.4 - 0.2, 0, 1) * (0.7 + 0.3 * noise(p, 0.8, seed + 2)))
    return Material(color=color)


def glowing(color, strength: float = 1.0) -> Material:
    return Material(color=tuple(np.asarray(color, float)), glow=strength)


def team(lightness: float = 0.7, pattern=None) -> Material:
    """Cloth or paint in the owner's colour (`lightness` sets its shade); `pattern(look)` can darken parts."""
    def color(l: Look):
        k = lightness if pattern is None else lightness * pattern(l)
        return np.repeat(np.asarray(k, float).reshape(-1, 1) * np.ones((len(l.p), 1)), 3, 1)
    return Material(color=color, team=True)


def metal(base=IRON, rust: float = 0.0, seed: int = 0, shine: float = 0.6) -> Material:
    base = np.asarray(base, float)

    def color(l: Look):
        c = mix(base, base * 1.25, noise(l.p, 1.0, seed))
        if rust:
            c = mix(c, RUST, np.clip((fbm(l.p, 2.0, seed + 3) - 0.5) * 3 * rust, 0, 1))
        return c
    return Material(color=color, shine=shine, gloss=30)
