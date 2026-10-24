"""Haunted houses: the pieces every Age of Halloween building is made of, so each one looks like a scary house (the
player asked, 2026-10-24: "make them in variation look like a scary house, with details; avoid pyramid-like
buildings; add spider webs from time to time").

`manor()` builds a house on a rectangle: storeys of the culture's walls with lit windows (some boarded up, some
with broken shutters, sills under them), the upper storey jutting out in the timbered cultures, a steep roof that
sags and leans (missing shingles, moss), dormers and an attic window on the gable end, a leaning chimney, a porch
with posts, steps and jack-o'-lanterns, and a turret in the culture's manner. Then `web()` (spider webs in corners),
`ravens()`, `fence()` (a broken picket fence), `dead_tree()` and the yard's gravestones.

Roofs: "gable" (steep, the ridge along x or y), "gambrel" (a barn's two-pitched roof), "mansard" (steep sides, a
flat top behind iron cresting), "curved" (the East Asian gable, eaves swept up at the corners), "flat" (behind a
crumbling parapet), "tiles" (a low terracotta gable, Day of the Dead). Never a four-sided pyramid.

Model +x and +y are the walls the camera sees, so doors, porches, windows and decorations face there.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import haunted as H
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part
from .haunted import box, hull

STOREY = 14.0
THREAD = S.Material(color=(0.86, 0.86, 0.9), glow=0.35, vivid=0.3)
DARK_WOOD = P.wood(P.hexrgb("#2e2018"), P.hexrgb("#100a06"), along=2, seed=31)
IRON = P.metal(P.hexrgb("#262228"), rust=0.4, seed=9)


def _rng(*key) -> np.random.Generator:
    import zlib
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def tilt(shapes: list, pivot, rot) -> list:
    """Shapes turned by `rot` (degrees about x, y, z) round `pivot`: a sagging, leaning roof or chimney."""
    T = S.affine(S.rotation(*rot), pivot) @ S.affine(None, tuple(-np.asarray(pivot, float)))
    for s in shapes:
        s.xf = T @ s.xf
    return shapes


def holey(m: S.Material, seed: int = 0) -> S.Material:
    """A roof with shingles missing (dark holes) and moss creeping over it."""
    def color(look):
        c = m.color(look) if callable(m.color) else np.broadcast_to(np.asarray(m.color, float), (len(look.p), 3))
        n = P.noise(look.p, 1.6, seed)
        hole = (n > 0.8) & (P.noise(look.p, 9.0, seed + 2) > 0.55)  # a few small patches, in places
        moss = np.clip((P.fbm(look.p, 6.0, seed + 1) - 0.58) * 3.0, 0, 0.6)
        c = P.mix(c, P.hexrgb("#2a3a1c"), moss)
        return np.where(hole[:, None], P.hexrgb("#0c0a0a"), c)
    return S.Material(color=color, bump=m.bump, shine=m.shine, gloss=m.gloss, vivid=m.vivid)


def _at(face: str, wall: float, u: float, z: float, out: float = 0.0) -> np.ndarray:
    """A point on a camera-facing wall: `face` +x (the wall x = wall, u along y) or +y (y = wall, u along x)."""
    return np.array([wall + out, u, z]) if face == "+x" else np.array([u, wall + out, z])


# --------------------------------------------------------------------------- small things

def web(corner, u, v, size: float = 8.0, seed: int = 0) -> list:
    """A spider's web in a corner: threads fanning out from `corner` between the directions u and v (a quarter
    circle), sagging rings across them, and sometimes its spider."""
    rng = _rng("web", seed)
    c = np.asarray(corner, float)
    u, v = np.asarray(u, float) / np.linalg.norm(u), np.asarray(v, float) / np.linalg.norm(v)
    angles = np.linspace(0.0, math.pi / 2, 6)
    out = []
    for a in angles:
        out.append(S.rod(tuple(c), tuple(c + (math.cos(a) * u + math.sin(a) * v) * size), 0.13, THREAD))
    for j in range(1, 5):
        r = size * j / 4.4 * rng.uniform(0.92, 1.05)
        pts = [c + (math.cos(a) * u + math.sin(a) * v) * r * (1 - 0.12 * math.sin(2 * a)) for a in angles]
        for a, b in zip(pts, pts[1:]):
            out.append(S.rod(tuple(a), tuple(b), 0.11, THREAD))
    if rng.random() < 0.5:
        sp = c + (u + v) * size * 0.32
        body = S.mat(P.hexrgb("#100c0e"), shine=0.5)
        out += [S.ball(tuple(sp), 0.9, body), S.ball(tuple(sp + (u + v) * 0.9), 0.6, body)]
    for s in out:
        s.casts = False
        s.ink = False
    return out


def ravens(points, seed: int = 0) -> list:
    """Ravens perched (on a ridge, a fence, a sign): a black body, a head, a beak, facing the camera's side."""
    m = S.mat(P.hexrgb("#0e0c12"), shine=0.6, gloss=30)
    beak = S.mat(P.hexrgb("#2a2420"))
    out = []
    for i, p in enumerate(points):
        p = np.asarray(p, float)
        d = np.array([0.6, 0.8, 0.0]) if i % 2 else np.array([0.8, 0.5, 0.0])
        out += [S.Ellipsoid(m, S._orient(tuple(p + (0, 0, 1.6)), (0, -20, 0)), radii=(1.0, 1.6, 1.2)),
                S.ball(tuple(p + d * 1.4 + (0, 0, 2.9)), 0.75, m),
                S.rod(tuple(p + d * 1.9 + (0, 0, 2.9)), tuple(p + d * 3.0 + (0, 0, 2.7)), 0.3, beak, 0.05),
                S.rod(tuple(p - d * 1.2 + (0, 0, 1.2)), tuple(p - d * 3.0 + (0, 0, 0.6)), 0.5, m, 0.15)]
    return out


def fence(a, b, height: float = 7.0, seed: int = 0, broken: float = 0.25) -> list:
    """A rotting picket fence from a to b: pointed pickets, some missing or leaning, two rails."""
    rng = _rng("fence", seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    wood = P.wood(P.hexrgb("#4a4440"), P.hexrgb("#1a1614"), along=2, seed=seed)
    out = [S.rod(tuple(a + (0, 0, height * 0.35)), tuple(b + (0, 0, height * 0.35)), 0.35, wood),
           S.rod(tuple(a + (0, 0, height * 0.75)), tuple(b + (0, 0, height * 0.75)), 0.35, wood)]
    n = max(2, int(L // 2.6))
    for i in range(n + 1):
        if rng.random() < broken * 0.6:
            continue
        p = a + d * i / n
        lean = rng.normal(0, 0.12 if rng.random() > broken else 0.45, 2)
        top = p + np.array([lean[0], lean[1], 1.0]) * height
        out.append(hull([p + (-0.6, -0.15, 0), p + (0.6, 0.15, 0), p + (-0.6, 0.15, 0), p + (0.6, -0.15, 0),
                         top + (-0.6, -0.15, -1.0), top + (0.6, 0.15, -1.0), top + (-0.6, 0.15, -1.0),
                         top + (0.6, -0.15, -1.0), top], wood))
    return out


def dead_tree(x, y, height: float = 60.0, variant: int = 0) -> Part:
    from .graveyard import dead_tree as tree
    t = tree(height, variant)
    return Part("dead_tree", offset=(x, y, 0.0)).add(t)


def graves(points, seed: int = 0) -> list:
    out = []
    for i, (x, y) in enumerate(points):
        out += H.gravestone(x, y, 0.0, 5.0 + (i % 3), i + seed, seed=i + seed)
    return out


# --------------------------------------------------------------------------- walls

def storey(k: H.Kit, x0, y0, x1, y1, z0, h, m=None, door=None, every=13.0, seed=0, plinth=0.0,
           boarded: float = 0.25, shutters: float = 0.4, faces=("+x", "+y")) -> list:
    """One storey: the walls with their painted windows (and door), and in 3D on the camera sides sills, broken
    shutters and boarded-up windows."""
    rng = _rng("storey", k.key, seed, x0, y0, z0)
    m = m or k.wall
    ops = H.wall_openings(k, x0, y0, x1, y1, z0, h, door, every, faces=faces)
    out = [box((x0, y0, z0), (x1, y1, z0 + h), H.facade(m, ops, k) if ops else m)]
    if plinth:
        out.append(box((x0 - 1.0, y0 - 1.0, max(0.0, z0 - plinth)), (x1 + 1.0, y1 + 1.0, z0 + 1.2), k.base))
    plank = P.wood(P.hexrgb("#4a3a30"), P.hexrgb("#1e1612"), along=1, seed=seed + 5)
    shut_m = P.wood(P.hexrgb("#3a3a40") if k.key in ("W", "G") else P.hexrgb("#3a2a22"), P.hexrgb("#121014"), along=2,
                    seed=seed + 6)
    for o in ops:
        if o.kind == "door":
            continue
        wall = x1 if o.face == "+x" else y1
        w = o.u1 - o.u0
        out.append(box(tuple(np.minimum(_at(o.face, wall, o.u0 - 0.6, o.z0 - 0.9), _at(o.face, wall, o.u1 + 0.6,
                                                                                         o.z0, 0.9))),
                       tuple(np.maximum(_at(o.face, wall, o.u0 - 0.6, o.z0 - 0.9), _at(o.face, wall, o.u1 + 0.6,
                                                                                         o.z0, 0.9))), k.trim))
        r = rng.random()
        if r < boarded:  # two planks nailed across in an X
            for (ua, za), (ub, zb) in (((o.u0 - 0.6, o.z0 + 0.6), (o.u1 + 0.6, o.z1 - 0.8)),
                                       ((o.u0 - 0.6, o.z1 - 0.8), (o.u1 + 0.6, o.z0 + 0.6))):
                pa, pb = _at(o.face, wall, ua, za, 0.5), _at(o.face, wall, ub, zb, 0.5)
                out.append(S.rod(tuple(pa), tuple(pb), 0.75, plank))
        elif r < boarded + shutters:  # shutters, one of them hanging off its top hinge
            for side, sgn in ((o.u0, -1), (o.u1, 1)):
                hang = rng.random() < 0.45
                a = side + sgn * 0.3
                b = side + sgn * (w * 0.5 + 0.3)
                lo = o.z0 + (rng.uniform(1.2, 3.0) if hang else 0.0)
                skew = rng.uniform(0.8, 2.0) * sgn if hang else 0.0
                pts = [_at(o.face, wall, a, o.z1 - 0.4, 0.4), _at(o.face, wall, b, o.z1 - 0.4, 0.4),
                       _at(o.face, wall, a + skew, lo, 0.7 + (0.6 if hang else 0)),
                       _at(o.face, wall, b + skew, lo, 0.7 + (0.6 if hang else 0))]
                pts += [p + (_at(o.face, 0, 0, 0, 0.4) - _at(o.face, 0, 0, 0)) for p in pts]
                out.append(hull(pts, shut_m))
    return out


# --------------------------------------------------------------------------- roofs

def _gable_ends(k, x0, x1, y0, y1, z, rise, axis, m) -> list:
    out = []
    if axis == "x":
        ym = (y0 + y1) / 2
        for x in (x0, x1):
            out.append(hull([(x - 0.3, y0, z), (x + 0.3, y0, z), (x - 0.3, y1, z), (x + 0.3, y1, z),
                             (x - 0.3, ym, z + rise - 0.4), (x + 0.3, ym, z + rise - 0.4)], m))
    else:
        xm = (x0 + x1) / 2
        for y in (y0, y1):
            out.append(hull([(x0, y - 0.3, z), (x0, y + 0.3, z), (x1, y - 0.3, z), (x1, y + 0.3, z),
                             (xm, y - 0.3, z + rise - 0.4), (xm, y + 0.3, z + rise - 0.4)], m))
    return out


def attic_window(k: H.Kit, x0, x1, y0, y1, z, rise, axis) -> list:
    """A round window high in the gable end that faces the camera, lit, with a dark frame and cross bars."""
    glow = S.mat(k.glow, glow=1.0)
    r = min(rise * 0.2, 3.2)
    if axis == "x":
        c = (x1 + 0.35, (y0 + y1) / 2, z + rise * 0.38)
        rot, bar_a, bar_b = (0, 90, 0), (0, r, 0), (0, 0, r)
    else:
        c = ((x0 + x1) / 2, y1 + 0.35, z + rise * 0.38)
        rot, bar_a, bar_b = (90, 0, 0), (r, 0, 0), (0, 0, r)
    c = np.asarray(c, float)
    pane = S.Ellipsoid(glow, S._orient(tuple(c), rot), radii=(r, r, 0.25))
    pane.casts = False
    return [pane, S.torus(tuple(c), r, 0.45, DARK_WOOD, rot=rot),
            S.rod(tuple(c - bar_a), tuple(c + bar_a), 0.25, DARK_WOOD),
            S.rod(tuple(c - bar_b), tuple(c + bar_b), 0.25, DARK_WOOD)]


def roof(k: H.Kit, x0, x1, y0, y1, z, kind="gable", axis="x", rise: Optional[float] = None, over=3.0, seed=0,
         crest: Optional[bool] = None) -> tuple[list, float]:
    """A roof over the rectangle at height z; returns (shapes, its top)."""
    m = holey(k.roof, seed)
    span = (y1 - y0) if axis == "x" else (x1 - x0)
    crest = k.crest if crest is None else crest
    if kind == "gable" or kind == "tiles" or kind == "curved":
        r = rise if rise is not None else min(span * (0.95 if kind == "gable" else 0.45 if kind == "tiles" else 0.6),
                                               34.0 + span * 0.15)  # steep, but a wide hall's roof no A-frame
        kk = H.Kit(**{**k.__dict__, "crest": crest and kind == "gable"})
        out = H.gable(kk, x0, x1, y0, y1, z, r, axis, over=over, m=m)
        out += attic_window(k, x0, x1, y0, y1, z, r, axis) if kind != "tiles" else []
        if kind == "curved":  # the eaves swept up at the four corners, a dark ridge board with horns at its ends
            horn = S.mat(P.hexrgb("#1a1414"), shine=0.4)
            if axis == "x":
                ym = (y0 + y1) / 2
                for x in (x0 - over, x1 + over):
                    for y in (y0 - over, y1 + over):
                        drop = r * over / (span / 2)
                        out.append(S.rod((x, y, z - drop + 0.6), (x + math.copysign(1.4, x - (x0 + x1) / 2),
                                                                  y + math.copysign(1.8, y - ym), z - drop + 4.4), 1.0,
                                         horn, 0.2))
                for x in (x0 - over, x1 + over):
                    out.append(S.rod((x, ym, z + r + 0.6), (x + math.copysign(2.0, x - (x0 + x1) / 2), ym,
                                                            z + r + 4.6), 1.0, horn, 0.25))
            else:
                xm = (x0 + x1) / 2
                for x in (x0 - over, x1 + over):
                    for y in (y0 - over, y1 + over):
                        drop = r * over / (span / 2)
                        out.append(S.rod((x, y, z - drop + 0.6), (x + math.copysign(1.8, x - xm),
                                                                  y + math.copysign(1.4, y - (y0 + y1) / 2),
                                                                  z - drop + 4.4), 1.0, horn, 0.2))
                for y in (y0 - over, y1 + over):
                    out.append(S.rod((xm, y, z + r + 0.6), (xm, y + math.copysign(2.0, y - (y0 + y1) / 2),
                                                            z + r + 4.6), 1.0, horn, 0.25))
        return out, z + r
    if kind == "gambrel":  # a barn: steep lower slopes, shallow upper ones
        r = rise if rise is not None else min(span * 0.75, 30.0 + span * 0.15)
        out = []
        if axis == "x":
            ym, hw = (y0 + y1) / 2, (y1 - y0) / 2
            prof = [(-hw - over, -over * 1.3), (-hw * 0.55, r * 0.62), (0, r)]
            for sgn in (1, -1):
                for (ya, za), (yb, zb) in zip(prof, prof[1:]):
                    pts = []
                    for x in (x0 - over, x1 + over):
                        for yy, zz in ((ya, za), (yb, zb)):
                            pts += [(x, ym - sgn * yy, z + zz), (x, ym - sgn * yy, z + zz + 1.4)]
                    out.append(hull(pts, m))
            for x in (x0, x1):
                pts = [(x + dx, ym + s * yy, z + zz) for dx in (-0.3, 0.3) for s in (1, -1)
                       for yy, zz in ((-hw, 0), (-hw * 0.55, r * 0.62), (0, r - 0.4))]
                out.append(hull(pts, k.upper))
        else:
            xm, hw = (x0 + x1) / 2, (x1 - x0) / 2
            prof = [(-hw - over, -over * 1.3), (-hw * 0.55, r * 0.62), (0, r)]
            for sgn in (1, -1):
                for (xa, za), (xb, zb) in zip(prof, prof[1:]):
                    pts = []
                    for y in (y0 - over, y1 + over):
                        for xx, zz in ((xa, za), (xb, zb)):
                            pts += [(xm - sgn * xx, y, z + zz), (xm - sgn * xx, y, z + zz + 1.4)]
                    out.append(hull(pts, m))
            for y in (y0, y1):
                pts = [(xm + s * xx, y + dy, z + zz) for dy in (-0.3, 0.3) for s in (1, -1)
                       for xx, zz in ((-hw, 0), (-hw * 0.55, r * 0.62), (0, r - 0.4))]
                out.append(hull(pts, k.upper))
        # a hay door high in the gable end facing the camera, lit
        if axis == "x":
            out.append(box((x1 + 0.1, (y0 + y1) / 2 - 3.0, z + r * 0.15), (x1 + 0.6, (y0 + y1) / 2 + 3.0, z + r * 0.55),
                           S.mat(k.glow, glow=0.9)))
        else:
            out.append(box(((x0 + x1) / 2 - 3.0, y1 + 0.1, z + r * 0.15), ((x0 + x1) / 2 + 3.0, y1 + 0.6, z + r * 0.55),
                           S.mat(k.glow, glow=0.9)))
        return out, z + r
    if kind == "mansard":  # steep sides with dormers, a flat top behind iron cresting (the Victorian haunted house)
        r = rise if rise is not None else 12.0
        inset = r * 0.32
        top = z + r
        pts = [(x, y, z) for x in (x0 - 1.5, x1 + 1.5) for y in (y0 - 1.5, y1 + 1.5)]
        pts += [(x, y, top) for x in (x0 + inset, x1 - inset) for y in (y0 + inset, y1 - inset)]
        out = [hull(pts, m), box((x0 - 2.2, y0 - 2.2, z - 0.4), (x1 + 2.2, y1 + 2.2, z + 0.8), k.trim),
               box((x0 + inset - 0.6, y0 + inset - 0.6, top), (x1 - inset + 0.6, y1 - inset + 0.6, top + 0.8), k.trim)]
        out += H.cresting((x0 + inset, y1 - inset, top + 0.8), (x1 - inset, y1 - inset, top + 0.8))
        out += H.cresting((x1 - inset, y0 + inset, top + 0.8), (x1 - inset, y1 - inset, top + 0.8))
        n = max(1, int((x1 - x0) // 22))
        for i in range(n):  # dormers on the two slopes the camera sees
            c = x0 + (x1 - x0) * (i + 0.5) / n
            out += dormer(k, "+y", c, y1 + 1.0 - inset * 0.5, z + r * 0.3, seed=seed + i)
        n = max(1, int((y1 - y0) // 22))
        for i in range(n):
            c = y0 + (y1 - y0) * (i + 0.5) / n
            out += dormer(k, "+x", c, x1 + 1.0 - inset * 0.5, z + r * 0.3, seed=seed + 10 + i)
        return out, top + 2.0
    if kind == "flat":  # behind a crumbling parapet, a few merlons fallen
        rng = _rng("flat", seed)
        out = [box((x0 - 0.8, y0 - 0.8, z), (x1 + 0.8, y1 + 0.8, z + 1.4), k.base)]
        for (a0, a1, b0, b1, along) in ((x0 - 0.8, x1 + 0.8, y1 - 0.4, y1 + 0.8, "x"),
                                        (x0 - 0.8, x1 + 0.8, y0 - 0.8, y0 + 0.4, "x"),
                                        (y0 - 0.8, y1 + 0.8, x1 - 0.4, x1 + 0.8, "y"),
                                        (y0 - 0.8, y1 + 0.8, x0 - 0.8, x0 + 0.4, "y")):
            n = max(2, int((a1 - a0) // 5.0))
            for i in range(n):
                u = a0 + (a1 - a0) * i / n
                hgt = 2.0 + (2.4 if i % 2 == 0 else 0.0) - (rng.uniform(0, 2.4) if rng.random() < 0.3 else 0.0)
                lo, hi = (u, b0, z + 1.4), (u + (a1 - a0) / n, b1, z + 1.4 + hgt)
                if along == "y":
                    lo, hi = (b0, u, z + 1.4), (b1, u + (a1 - a0) / n, z + 1.4 + hgt)
                out.append(box(lo, hi, k.upper))
        return out, z + 5.0
    raise ValueError(kind)


def dormer(k: H.Kit, face: str, u: float, wall: float, z: float, w: float = 6.0, seed: int = 0) -> list:
    """A dormer window standing out of a roof slope on the camera's side: a little lit window under its own
    gable, its back sunk into the roof."""
    if face == "+y":
        x0, x1, y0, y1 = u - w / 2, u + w / 2, wall - 8.0, wall
        ops = [H.Opening("+y", u - w * 0.28, u + w * 0.28, z + 1.4, z + 6.4, k.window)]
        axis = "y"
    else:
        x0, x1, y0, y1 = wall - 8.0, wall, u - w / 2, u + w / 2
        ops = [H.Opening("+x", u - w * 0.28, u + w * 0.28, z + 1.4, z + 6.4, k.window)]
        axis = "x"
    out = [box((x0, y0, z - 3.0), (x1, y1, z + 7.5), H.facade(k.upper, ops, k))]
    out += H.gable(H.Kit(**{**k.__dict__, "crest": False}), x0, x1, y0, y1, z + 7.5, w * 0.6, axis, over=1.0,
                   m=holey(k.roof, seed + 3), thick=0.9)
    return out


def chimney(x, y, z0, top, lean=(3.0, -2.0, 0.0), seed=0) -> list:
    """A brick chimney leaning out of true, a cap and a black mouth."""
    brick = P.bricks(P.hexrgb("#5a3a30"), P.hexrgb("#24140f"), 2.4, 1.2, moss=0.3, seed=seed)
    out = [box((x - 3.0, y - 3.0, z0), (x + 3.0, y + 3.0, top), brick),
           box((x - 3.8, y - 3.8, top), (x + 3.8, y + 3.8, top + 1.4), brick),
           box((x - 2.0, y - 2.0, top + 1.3), (x + 2.0, y + 2.0, top + 1.5), S.mat(P.hexrgb("#0c0a0a")))]
    return tilt(out, (x, y, z0), lean)


# --------------------------------------------------------------------------- porch, turret

def porch(k: H.Kit, face: str, a0: float, a1: float, wall: float, depth: float = 9.0, h: float = 12.0,
          z0: float = 0.0, seed: int = 0, webbed: bool = True) -> list:
    """A sagging porch along a camera-facing wall: a plank deck, turned posts, a railing with a gap for the steps,
    a lean-to roof, a hanging lantern, jack-o'-lanterns on the steps and a web in a corner."""
    rng = _rng("porch", seed)
    deck_m = P.wood(P.hexrgb("#3e3028"), P.hexrgb("#16100c"), along=0 if face == "+y" else 1, planks=1.6, seed=seed)
    post_m = k.trim

    def pt(u, out, z):
        return _at(face, wall, u, z, out)
    out = [box(tuple(np.minimum(pt(a0, 0, z0), pt(a1, depth, z0 + 2.4))), tuple(np.maximum(pt(a0, 0, z0),
                                                                                             pt(a1, depth, z0 + 2.4))),
               deck_m)]
    mid = (a0 + a1) / 2
    posts = [a0 + 1.0, a1 - 1.0] + ([mid - 6.0, mid + 6.0] if a1 - a0 > 34 else [])
    for u in posts:
        out.append(S.turned(tuple(pt(u, depth - 1.0, z0 + 2.4)), [(0, 0), (0.9, 0), (0.6, 1.5), (0.55, h - 2.0),
                                                                   (0.9, h - 1.0), (0.9, h), (0, h)], post_m))
    # the railing, open in front of the steps
    for ua, ub in ((a0 + 1.0, mid - 5.0), (mid + 5.0, a1 - 1.0)):
        if ub - ua < 2:
            continue
        out.append(S.rod(tuple(pt(ua, depth - 1.0, z0 + 6.6)), tuple(pt(ub, depth - 1.0, z0 + 6.6)), 0.35, post_m))
        for i in range(int((ub - ua) // 2.2)):
            u = ua + 1.1 + i * 2.2
            if rng.random() < 0.15:
                continue
            out.append(S.rod(tuple(pt(u, depth - 1.0, z0 + 2.4)), tuple(pt(u, depth - 1.0, z0 + 6.4)), 0.22, post_m))
    # steps down from the gap
    for i, (d0, d1, zz) in enumerate(((depth, depth + 2.6, z0 + 1.6), (depth + 2.6, depth + 5.2, z0 + 0.8))):
        out.append(box(tuple(np.minimum(pt(mid - 4.5, d0, z0), pt(mid + 4.5, d1, zz))),
                       tuple(np.maximum(pt(mid - 4.5, d0, z0), pt(mid + 4.5, d1, zz))), deck_m))
    out += PK.lantern(tuple(pt(mid - 6.5, depth + 1.6, z0 + 1.6)), 1.5, seed=seed)
    out += PK.lantern(tuple(pt(mid + 6.0, depth + 3.8, z0 + 0.8)), 1.3, seed=seed + 1)
    # the lean-to roof, sagging a little
    roof_m = holey(k.roof, seed + 7)
    hi, lo = z0 + h + 4.0, z0 + h + 0.4
    pts = [pt(a0 - 1.0, -0.5, hi), pt(a1 + 1.0, -0.5, hi), pt(a0 - 1.0, depth + 1.5, lo), pt(a1 + 1.0, depth + 1.5, lo)]
    pts += [p + (0, 0, 1.0) for p in pts]
    out += tilt([hull(pts, roof_m)], pt(a1, 0, hi), (0.0, 0.0, 0.0) if face == "+y" else (0.0, 0.0, 0.0))
    lamp = pt(mid + 2.5, depth - 1.5, z0 + h - 1.0)
    out += [S.rod(tuple(lamp + (0, 0, 1.0)), tuple(lamp + (0, 0, 3.0)), 0.12, IRON),
            S.Ellipsoid(S.mat(k.glow, glow=1.0), S._orient(tuple(lamp)), radii=(0.9, 0.9, 1.2))]
    if webbed:
        corner = pt(a1 - 1.0, depth - 1.0, z0 + h - 0.2)
        out += web(corner, pt(-1.0, 0, 0) - pt(0, 0, 0), (0, 0, -1.0), 7.0, seed)
    return out


def turret(k: H.Kit, cx, cy, r, z0, h, seed=0, flag=True) -> list:
    """The culture's turret: round under a bent witch's hat (G, W), square with a needle spire and pinnacles (E), a
    minaret with an onion dome (M), a lantern tower under a curved roof (F), a painted belfry with a bell (X)."""
    if k.key in ("G", "W", "E", "M"):
        out = H.tower(k, cx, cy, r, z0, h, flag=flag, seed=seed)
        if k.key in ("G", "W"):
            out += web((cx + r * 0.7, cy + r * 0.7, z0 + h - 0.5), (0.7, -0.7, 0), (0, 0, -1), r * 1.1, seed + 3)
        return out
    if k.key == "F":
        out = storey(k, cx - r, cy - r, cx + r, cy + r, z0, h * 0.75, every=r * 3, seed=seed, boarded=0, shutters=0)
        out += storey(k, cx - r * 0.8, cy - r * 0.8, cx + r * 0.8, cy + r * 0.8, z0 + h * 0.75, h * 0.25,
                      m=k.upper, every=r, seed=seed + 1, boarded=0, shutters=0)
        rf, top = roof(k, cx - r * 0.8, cx + r * 0.8, cy - r * 0.8, cy + r * 0.8, z0 + h, "curved", "x", over=r * 0.5,
                       seed=seed)
        out += rf + H.lanterns_hanging([(cx + r + 1.5, cy, z0 + h * 0.7), (cx, cy + r + 1.5, z0 + h * 0.7)], k)
        if flag:
            out += H.team_flag((cx, cy, top - 1.0), 10.0, seed)
        return out
    # X: a painted belfry, its bell in an arched opening, a little dome and a cross of iron
    out = [box((cx - r, cy - r, z0), (cx + r, cy + r, z0 + h), k.wall),
           box((cx - r - 0.6, cy - r - 0.6, z0 + h), (cx + r + 0.6, cy + r + 0.6, z0 + h + 1.2), k.base)]
    bz = z0 + h + 1.2
    ops = [H.Opening(f, -r * 0.5, r * 0.5, bz + 1.0, bz + r * 1.6, "round") for f in ("+x", "+y")]
    local = H.facade(k.upper, ops, H.Kit(**{**k.__dict__, "glow": P.hexrgb("#1a1014")}))
    out.append(box((cx - r * 0.9, cy - r * 0.9, bz), (cx + r * 0.9, cy + r * 0.9, bz + r * 1.9), local))
    out.append(S.turned((cx, cy + r * 0.92, bz + r * 0.5), [(0, 0), (r * 0.42, 0), (r * 0.3, r * 0.5), (0.1, r * 0.75)],
                        P.metal(P.GOLD, rust=0.4)))
    out += H.dome(cx, cy, bz + r * 1.9, r * 0.8, k.roof, finial=P.hexrgb("#2a2628"))
    if flag:
        out += H.team_flag((cx, cy, bz + r * 3.0), 10.0, seed)
    return out


# --------------------------------------------------------------------------- the house

ROOFS = {"G": "gable", "W": "gable", "E": "gable", "M": "flat", "F": "curved", "X": "tiles"}


def manor(k: H.Kit, x0, y0, x1, y1, floors: int = 2, axis: str = "x", door_face: str = "+y",
          door_at: Optional[float] = None, roof_kind: Optional[str] = None, porch_at: bool = True,
          tower: Optional[tuple] = None, dormers: int = 1, chimney_at: Optional[tuple] = None, lean: float = 1.6,
          seed: int = 0, storey_h: float = STOREY, webs: bool = True, z0: float = 0.0) -> tuple[list, float]:
    """A haunted house on the rectangle; returns (shapes, roof top). `tower` is (x, y, radius, height)."""
    rng = _rng("manor", k.key, k.age, seed, x0, y0)
    roof_kind = roof_kind or ROOFS[k.key]
    if door_at is None:
        door_at = (x0 + x1) / 2 if door_face == "+y" else (y0 + y1) / 2
    out = [box((x0 - 2.0, y0 - 2.0, z0), (x1 + 2.0, y1 + 2.0, z0 + 2.5), k.base)]  # the foundation
    z = z0 + 2.5
    jetty = 1.6 if k.key in ("W", "G") else 0.0
    door_h = min(storey_h * 0.82, 12.0)
    for f in range(floors):
        grow = jetty * f
        fx0, fy0, fx1, fy1 = x0, y0, x1 + grow, y1 + grow
        m = k.wall if f == 0 else k.upper
        door = (door_face, door_at, 6.5, door_h) if f == 0 else None
        out += storey(k, fx0, fy0, fx1, fy1, z, storey_h, m=m, door=door, seed=seed + f)
        z += storey_h
        out.append(box((fx0 - 0.6, fy0 - 0.6, z - 0.8), (fx1 + 0.8, fy1 + 0.8, z + 0.4), k.trim))  # a floor band
    grow = jetty * (floors - 1)
    rx1, ry1 = x1 + grow, y1 + grow
    span = (ry1 - y0) if axis == "x" else (rx1 - x0)
    rf, top = roof(k, x0, rx1, y0, ry1, z, roof_kind, axis, seed=seed, over=3.0)
    if roof_kind in ("gable", "tiles", "curved"):
        hw = span / 2
        rise = top - z
        for i in range(dormers):  # dormers on the camera-side slope
            if axis == "x":
                u = x0 + (rx1 - x0) * (i + 1) / (dormers + 1)
                wall = (y0 + ry1) / 2 + hw * 0.55
                dz = z + rise * (1 - 0.55) - 3.0
                rf += dormer(k, "+y", u, wall, dz, seed=seed + 20 + i)
            else:
                u = y0 + (ry1 - y0) * (i + 1) / (dormers + 1)
                wall = (x0 + rx1) / 2 + hw * 0.55
                dz = z + rise * (1 - 0.55) - 3.0
                rf += dormer(k, "+x", u, wall, dz, seed=seed + 20 + i)
        # the roof sags and leans: turned a little round its eave on the far side
        pivot = ((x0 + rx1) / 2, (y0 + ry1) / 2, z)
        sag = (rng.uniform(-1, 1) * lean, rng.uniform(-1, 1) * lean, rng.uniform(-1, 1) * lean * 0.6)
        rf = tilt(rf, pivot, sag)
        ridge_y = (y0 + ry1) / 2 if axis == "x" else None
        if rng.random() < 0.6:
            if axis == "x":
                out += ravens([(x0 + (rx1 - x0) * rng.uniform(0.3, 0.7), ridge_y, top + 0.6)], seed)
            else:
                out += ravens([((x0 + rx1) / 2, y0 + (ry1 - y0) * rng.uniform(0.3, 0.7), top + 0.6)], seed)
    out += rf
    if chimney_at is not None:
        cx, cy = chimney_at
        out += chimney(cx, cy, z - 6.0, top + 4.0, (rng.uniform(-4, 4), rng.uniform(-4, 4), 0), seed)
    if porch_at:
        if door_face == "+y":
            out += porch(k, "+y", max(x0 + 2.0, door_at - 16.0), min(x1 - 2.0, door_at + 16.0), y1, seed=seed,
                         webbed=webs)
        else:
            out += porch(k, "+x", max(y0 + 2.0, door_at - 16.0), min(y1 - 2.0, door_at + 16.0), x1, seed=seed,
                         webbed=webs)
    if tower is not None:
        tx, ty, tr, th = tower
        out += turret(k, tx, ty, tr, z0, th, seed=seed)
    if webs and rng.random() < 0.75:  # a web under the eaves at the corner the camera sees
        out += web((rx1 + 0.3, ry1 + 0.3, z - 0.3), (-1.0, 0, 0) if rng.random() < 0.5 else (0, -1.0, 0), (0, 0, -1.0),
                   rng.uniform(6.0, 9.0), seed + 50)
    if k.key == "E":  # gargoyles hunched on the front corners of the roof
        out += gargoyles(k, [(rx1 + 1.0, ry1 + 1.0, z + 0.4), (rx1 + 1.0, y0 - 1.0, z + 0.4)], seed)
    return out, top


def gargoyles(k: H.Kit, points, seed=0) -> list:
    """Stone gargoyles crouching on corners: a hunched body, a horned head leaning out, little wings, red eyes."""
    stone = P.mottled(P.hexrgb("#5a5660"), P.hexrgb("#24222a"), 1.2, 0.5, seed)
    out = []
    for p in points:
        p = np.asarray(p, float)
        d = np.array([0.7, 0.7, 0.0])
        head = p + d * 2.6 + (0, 0, 3.4)
        out += [S.Ellipsoid(stone, S._orient(tuple(p + (0, 0, 1.8))), radii=(1.6, 1.6, 2.0)),
                S.Ellipsoid(S.mat(P.hexrgb("#ff2410"), glow=1.0), S._orient(tuple(head + d * 1.1 + (0, 0, 0.3))),
                            radii=(0.9, 0.9, 0.3)),
                S.ball(tuple(head), 1.3, stone),
                S.rod(tuple(head + (0.4, -0.4, 0.8)), tuple(head + (0.2, -1.2, 2.4)), 0.35, stone, 0.05),
                S.rod(tuple(head + (-0.4, 0.4, 0.8)), tuple(head + (-1.2, 0.2, 2.4)), 0.35, stone, 0.05),
                hull([p + (0, 0, 3.0), p + (-0.6, -1.8, 4.6), p + (-2.2, -0.4, 4.2), p + (-0.4, -0.4, 1.4)], stone)]
    return out


def yard(k: H.Kit, x0, y0, x1, y1, seed=0, tree: bool = True) -> tuple[list, list]:
    """A yard in front of a house (the camera's side of the rectangle): a broken fence along its edge, a couple of
    gravestones, and maybe a dead tree; returns (shapes, child parts)."""
    rng = _rng("yard", k.key, seed)
    out = fence((x0, y1, 0.0), (x1 - 10.0, y1, 0.0), 6.0, seed) + fence((x1, y0, 0.0), (x1, y1 - 10.0, 0.0), 6.0,
                                                                         seed + 1)
    out += graves([(x1 - 6.0 - rng.uniform(0, 8), y1 - 8.0), (x1 - 14.0 - rng.uniform(0, 6), y1 - 14.0)], seed)
    parts = []
    if tree:
        parts.append(dead_tree(x0 + 6.0, y1 - 6.0, rng.uniform(40, 58), seed % 5))
    if rng.random() < 0.6:
        out += web((x1, y1 - 10.0, 6.0), (0, 1.0, 0), (0, 0, -1.0), 5.0, seed + 9)
    return out, parts
