"""Haunted buildings, in five haunted cultures (one per building set):

    W  West European    haunted Victorian: crooked painted clapboard, steep shingle roofs, witch-hat turrets
    E  Central/North    Transylvanian gothic: dark stone, pointed arches, spires, vampire bats
    M  Middle Eastern   mummy tombs: sandstone, hieroglyph bands, sarcophagi, scarab gold
    F  East Asian       ghost lanterns and yokai shrines: dark timber, paper lanterns, curled roofs
    X  Meso-American    Day of the Dead: bright painted walls, marigolds, sugar skulls, papel picado

Buildings stand with their footprint centre on the sprite's hotspot, seen from the fixed building view: model +x
and +y are the two walls facing the camera (heading voxel.BUILDING_HEADING). One tile is voxel.TILE (about 45) units.
"""
from __future__ import annotations

import math

import numpy as np

from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part

TILE = 96 / (1.5 * 2 ** 0.5)


# --------------------------------------------------------------------------- materials

def siding(paint_rgb=P.hexrgb("#6c6378"), wood_rgb=P.WOOD_DARK, board: float = 2.2, peel: float = 0.5, seed=0
           ) -> S.Material:
    """Painted clapboard: horizontal boards, each shadowed under the one above, the paint faded and peeling to the
    dark wood beneath."""
    paint_rgb, wood_rgb = np.asarray(paint_rgb, float), np.asarray(wood_rgb, float)

    def color(look):
        p = look.p
        z = p[:, 2]
        f = (z / board) % 1.0
        c = P.mix(paint_rgb * 0.62, paint_rgb, np.clip(f * 1.6, 0, 1))  # the lap shadow under each board
        c = P.mix(c, paint_rgb * 0.8, 0.35 * P.fbm(p, 4.0, seed))
        flake = np.clip((P.fbm(p * np.array([1, 1, 2.5]), 3.0, seed + 7) - 0.62) * 6 * peel, 0, 1)
        return P.mix(c, wood_rgb, flake)
    return S.Material(color=color)


def window(arch: bool = True, glow=P.hexrgb("#ffb347"), seed: int = 0) -> S.Material:
    """A lit window: warm light behind panes divided by dark mullions, a little flicker across the glass."""
    glow = np.asarray(glow, float)

    def panes(p):
        u = np.where(np.abs(p[:, 0]) > np.abs(p[:, 1]), p[:, 0], p[:, 1])
        return (np.abs((u + 0.0) % 2.4 - 1.2) < 0.22) | (np.abs((p[:, 2]) % 3.0 - 1.5) < 0.22)

    def color(look):
        m = panes(look.p)
        lit = P.mix(glow, glow * 0.75, P.noise(look.p, 2.0, seed))
        return np.where(m[:, None], P.hexrgb("#1e1418"), lit)

    def glow_f(look):
        return np.where(panes(look.p), 0.0, 1.0)
    return S.Material(color=color, glow=glow_f)


def _moved(shapes, by):
    for s in shapes:
        s.xf = S.affine(None, tuple(np.asarray(by, float))) @ s.xf
    return shapes


def gothic_window(x0, x1, z0, z1, depth, face: str, at: float, frame_m: S.Material, glass: S.Material) -> list:
    """A pointed-arch window on a wall facing +x or +y (`at` = the wall's coordinate): glass inset in a dark frame."""
    w = x1 - x0
    zs = z1 - w * 0.55

    def pts(inset, out):
        a, b = x0 + inset, x1 - inset
        top = z1 - inset * 0.8
        mid = (a + b) / 2
        prof = [(a, z0 + inset), (b, z0 + inset), (b, zs), (mid + (b - a) * 0.2, top - (top - zs) * 0.35),
                (mid, top), (mid - (b - a) * 0.2, top - (top - zs) * 0.35), (a, zs)]
        res = []
        for u, z in prof:
            for d in (-depth, out):
                res.append((at + d, u, z) if face == "x" else (u, at + d, z))
        return res
    return [S.hull(pts(0.0, 0.6), frame_m), S.hull(pts(0.7, 0.9), glass)]


# --------------------------------------------------------------------------- W: the haunted Victorian

def _gable_roof(x0, x1, y0, y1, z0, rise, overhang, m_roof, m_end, thick=1.2) -> list:
    """A gable roof along x: two shingled slopes and the two triangular ends, with an iron crest on the ridge."""
    ym = (y0 + y1) / 2
    ridge = z0 + rise
    out = []
    for sgn in (-1, 1):
        ye = (y1 + overhang) if sgn > 0 else (y0 - overhang)
        drop = rise * overhang / ((y1 - y0) / 2)
        pts = []
        for x in (x0 - overhang, x1 + overhang):
            pts += [(x, ym, ridge + thick), (x, ym, ridge), (x, ye, z0 - drop), (x, ye, z0 - drop + thick)]
        out.append(S.hull(pts, m_roof))
    for x in (x0, x1):
        out.append(S.hull([(x - 0.2, y0, z0), (x + 0.2, y0, z0), (x - 0.2, y1, z0), (x + 0.2, y1, z0),
                           (x - 0.2, ym, ridge - 0.3), (x + 0.2, ym, ridge - 0.3)], m_end))
    iron = P.metal(rust=0.4, seed=9)
    out.append(S.rod((x0 - overhang, ym, ridge + thick), (x1 + overhang, ym, ridge + thick), 0.35, iron))
    for k, x in enumerate(np.linspace(x0, x1, 7)):
        h = 2.6 if k % 2 == 0 else 1.6
        out += [S.rod((x, ym, ridge + thick), (x, ym, ridge + thick + h), 0.2, iron),
                S.hull([(x - 0.5, ym, ridge + thick + h), (x + 0.5, ym, ridge + thick + h),
                        (x, ym - 0.5, ridge + thick + h), (x, ym + 0.5, ridge + thick + h),
                        (x, ym, ridge + thick + h + 1.4)], iron)]
    return out


def _witch_turret(cx, cy, r, z0, wall_h, roof_h, m_wall, m_roof, glass, seed=0) -> list:
    """A round turret with a tall, slightly bent witch-hat roof and a pennant in the owner's colour."""
    out = [S.rod((cx, cy, z0), (cx, cy, z0 + wall_h), r, m_wall)]
    z1 = z0 + wall_h
    prof = [(r + 2.2, 0), (r + 2.0, 0.8), (r * 0.9, roof_h * 0.18), (r * 0.55, roof_h * 0.45),
            (r * 0.26, roof_h * 0.75), (0.15, roof_h)]
    roof = S.turned((cx, cy, z1), [(0, 0)] + prof, m_roof, rot=(4, -6, 0))
    out.append(roof)
    tip = np.array([cx, cy, z1]) + S.rotation(4, -6, 0) @ np.array([0, 0, roof_h])
    iron = P.metal(rust=0.3)
    out.append(S.rod(tip - (0, 0, 1.5), tip + (0, 0, 7), 0.25, iron))
    flag = S.hull([tip + (0, 0, 6.6), tip + (0, 0, 3.6), tip + (6.5, 0.8, 5.4), tip + (0.0, 0.25, 5.0),
                   tip + (0, -0.25, 5.0)], P.team(0.68))
    out.append(flag)
    for a in (35, 100):  # two narrow lit windows round the turret's front
        ang = math.radians(a)
        wx, wy = cx + math.cos(ang) * (r - 0.4), cy + math.sin(ang) * (r - 0.4)
        win = S.block((-0.6, -1.1, -2.6), (0.6, 1.1, 2.6), glass, rot=(0, 0, a), at=(wx, wy, z0 + wall_h * 0.62))
        out.append(win)
    return out


def victorian_house(age: int = 2, variant: int = 0, seed: int = 0) -> Part:
    """A crooked haunted cottage: painted clapboard on a stone base, a steep shingled gable roof with iron cresting,
    a witch-hat turret at the corner, a leaning chimney, lit gothic windows and jack-o'-lanterns by the door."""
    rng = np.random.default_rng(seed + variant)
    paint_rgb = [P.hexrgb("#5a5462"), P.hexrgb("#4c5658"), P.hexrgb("#5e4c54")][variant % 3]
    wall = siding(paint_rgb, peel=0.9, seed=seed)
    base = P.bricks(P.hexrgb("#7d7884"), P.hexrgb("#4f4a58"), w=4.5, h=2.2, moss=0.9, seed=seed)
    roof = P.shingles(P.hexrgb("#4a3560"), P.hexrgb("#251a33"), seed=seed, axes=(0, 2))
    dark_wood = P.wood(P.hexrgb("#3c2a24"), P.hexrgb("#1f1512"), along=2)
    glass = window(seed=seed)
    x0, x1, y0, y1 = -24.0, 18.0, -20.0, 16.0
    zb, zw = 3.0, 24.0  # base and wall heights
    out = [S.block((x0 - 1, y0 - 1, 0), (x1 + 1, y1 + 1, zb), base)]
    body = S.block((x0, y0, zb), (x1, y1, zb + zw), wall, rot=(0, 0, 1.5))  # a little crooked
    out.append(body)
    out += _gable_roof(x0, x1, y0, y1, zb + zw, 20.0, 3.0, roof, wall)
    # windows: two on the front (+y), one on the side (+x), one in the gable
    for u in (-14.0, 4.0):
        out += gothic_window(u, u + 6, zb + 9, zb + 20, 0.4, "y", y1 + 0.3, dark_wood, glass)
    out += gothic_window(-6, 0, zb + 9, zb + 20, 0.4, "x", x1 + 0.3, dark_wood, glass)
    out += gothic_window(-3.5, 1.5, zb + zw + 3, zb + zw + 12, 0.3, "x", x1 + 0.6, dark_wood, glass)
    # the door with a little porch roof and steps; lanterns either side
    out.append(S.block((-6.5, y1 - 0.2, zb), (-1.0, y1 + 0.6, zb + 13), dark_wood))
    out.append(S.ball((-1.9, y1 + 0.9, zb + 6.5), 0.45, P.metal(P.GOLD)))
    out.append(S.block((-9.0, y1, zb + 13.4), (1.5, y1 + 5.5, zb + 14.4), roof, rot=(-14, 0, 0)))
    for k in range(2):
        out.append(S.block((-8.5, y1 + 0.6 + k * 1.8, 0), (1.0, y1 + 2.4 + k * 1.8, zb - k * 1.4), base))
    for x in (-10.5, 3.0):
        out += PK.lantern((x, y1 + 2.8, 0.0), 2.3, seed=seed + int(x))
    # the turret on the front corner and the chimney at the back
    out += _witch_turret(x1 + 1.0, y1 + 1.0, 6.5, 0.0, zb + zw + 8, 28.0, wall, roof, glass, seed)
    chim = P.bricks(P.hexrgb("#6e4a3e"), P.hexrgb("#3e2620"), w=2.4, h=1.2, seed=seed + 2)
    out.append(S.block((-19, -12, zb + zw + 4), (-14, -7, zb + zw + 24), chim, rot=(0, -5, 0)))
    out.append(S.block((-19.8, -12.8, zb + zw + 23), (-13.2, -6.2, zb + zw + 25), chim, rot=(0, -5, 0)))
    if age >= 3:  # a lightning rod and a weathervane bat
        pass
    return Part("house", boxes=out)
