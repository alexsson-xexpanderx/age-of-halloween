"""Smooth models and their renderer: curved shapes, soft cartoon light, ink outlines and smoothed edges.

Age of Minecraft drew everything in boxes, one texel per screen pixel. Age of Halloween's models are made of
curved shapes that rays hit exactly:

    Ellipsoid   spheres, eggs, heads, pumpkins (and halves of them: shapes can be cut by planes)
    Frustum     cylinders and cones, open or capped: bones, posts, chimneys, spires
    Lathe       a profile turned round an axis: domes, vases, urns, tower roofs, rings, tori
    Convex      flat-faced solids from planes: boxes, wedges, roofs, blades, gravestones

A shape sits in a `geometry.Part` (in its `boxes` list, next to any old boxes), so the animation rigs pose them
exactly as they posed the Minecraft models, and `render` returns the same `Frame` the SLP export expects.

Each screen pixel is sampled `SAMPLES` x `SAMPLES` times; a pixel is drawn when most of its samples hit the model,
in the average colour of those, so edges are smooth without ever being half transparent (the game's sprites have
no partial transparency). Ink lines are drawn where the depth jumps (an arm in front of a body, a roof's eave) and
round the outside, as in a cartoon. Light: a warm key light from the upper left, a purple-blue sky above, a cool
moonlit rim on the far side, and materials that glow (jack-o'-lantern faces, windows, eyes) ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from .geometry import Box, PlacedBox, Pose, affine, place, rotation
from .render import (LIGHT, OUTLINE, PLAYER, SHADOW, SOLID, SUN, TRANSPARENT, Camera, Frame, _corners, _FACE,
                     _NORMAL, _slabs, _uv, model_matrix)

SAMPLES = 3  # samples per pixel along each axis
KEY = LIGHT  # the warm key light: upper left, in front
KEY_RGB = np.array([1.0, 0.93, 0.80])
SKY_RGB = np.array([0.50, 0.46, 0.70])  # the fill from above: a dusky purple sky
GROUND_RGB = np.array([0.36, 0.30, 0.24])  # the fill from below: warm earth
RIM_RGB = np.array([0.55, 0.75, 0.80])  # moonlight on the far edges
WRAP = 0.35  # how far the key light wraps round (soft, cartoon-like)
INK = np.array([0.10, 0.06, 0.09])  # the outline colour: a very dark purple-brown
INK_DEPTH = 1.6  # model units: a depth step at least this big gets an ink line
INF = np.inf


# --------------------------------------------------------------------------- materials

@dataclass
class Look:
    """What a hit point looks like, for a material's colour function: local point and normal (in the shape's own
    frame), the world normal, and which face was hit ("surface" or "cut")."""
    p: np.ndarray
    n: np.ndarray
    nw: np.ndarray
    face: np.ndarray


Colour = Union[tuple, Callable[[Look], np.ndarray]]


@dataclass
class Material:
    color: Colour = (0.7, 0.7, 0.7)  # RGB 0..1, or a function of the Look giving (n, 3)
    glow: Union[float, Callable[[Look], np.ndarray]] = 0.0  # 1 = shines by itself (ignores light)
    team: Union[bool, Callable[[Look], np.ndarray]] = False  # drawn in the player's colour
    shine: float = 0.0  # specular highlight strength
    gloss: float = 24.0  # specular tightness
    bump: Optional[Callable[[Look], np.ndarray]] = None  # perturbed local normals
    cut: Optional["Material"] = None  # what a clip plane's face shows (the inside), default: this material

    def evaluate(self, look: Look):
        n = len(look.p)
        rgb = self.color(look) if callable(self.color) else np.broadcast_to(np.asarray(self.color, float), (n, 3))
        glow = self.glow(look) if callable(self.glow) else np.full(n, float(self.glow))
        team = self.team(look) if callable(self.team) else np.full(n, bool(self.team))
        return np.asarray(rgb, float), np.asarray(glow, float), np.asarray(team, bool)


def mat(color=(0.7, 0.7, 0.7), **kw) -> Material:
    return Material(color=color, **kw)


# --------------------------------------------------------------------------- shapes

def _orient(at=(0, 0, 0), rot=(0, 0, 0)) -> np.ndarray:
    return affine(rotation(*rot), at)


def axis_frame(a, b) -> np.ndarray:
    """A placement whose +z runs from a to b (for frusta and lathes between two points)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = b - a
    length = np.linalg.norm(z)
    z = z / length if length > 1e-9 else np.array([0, 0, 1.0])
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, a
    return M


@dataclass
class Shape:
    material: Material
    xf: np.ndarray = field(default_factory=lambda: np.eye(4))  # shape -> part
    clips: list = field(default_factory=list)  # (normal, k): keep n . x <= k (in shape coordinates)
    ink: bool = True  # takes part in ink lines

    def clip(self, normal, k) -> "Shape":
        n = np.asarray(normal, float)
        self.clips.append((n / np.linalg.norm(n), float(k)))
        return self

    # subclasses: corners() -> (m, 3) bounding points; span(o, d) -> (t_in, t_out, n_in) for convex shapes, or
    # nearest(o, d) -> (t, n) for the others
    convex = True

    def trace(self, o: np.ndarray, d: np.ndarray):
        """Entry distance (inf when missed), entry normal and whether a clip plane was hit first."""
        if self.convex:
            t_in, t_out, n_in = self.span(o, d)
        else:
            t_in, n_in = self.nearest(o, d)
            t_out = np.full(len(o), INF)
        cut = np.zeros(len(o), bool)
        for n, k in self.clips:
            den = float(n @ d)
            num = k - o @ n
            if abs(den) < 1e-12:
                t_in = np.where(num < 0, INF, t_in)
                continue
            t = num / den
            if den < 0:  # entering the kept side
                later = t > t_in
                t_in = np.where(later, t, t_in)
                n_in = np.where(later[:, None], n, n_in)
                cut |= later
            else:
                t_out = np.minimum(t_out, t)
        miss = ~(t_in <= t_out) | ~np.isfinite(t_in)
        t_in = np.where(miss, INF, t_in)
        return t_in, n_in, cut & ~miss


@dataclass
class Ellipsoid(Shape):
    radii: tuple = (1.0, 1.0, 1.0)

    def corners(self):
        r = np.asarray(self.radii, float)
        return np.array([[sx * r[0], sy * r[1], sz * r[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    def span(self, o, d):
        r = np.asarray(self.radii, float)
        o2, d2 = o / r, d / r
        a = float(d2 @ d2)
        b = 2 * (o2 @ d2)
        c = (o2 * o2).sum(1) - 1
        disc = b * b - 4 * a * c
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0))
        t0 = np.where(ok, (-b - sq) / (2 * a), INF)
        t1 = np.where(ok, (-b + sq) / (2 * a), -INF)
        p = o + np.where(ok, t0, 0)[:, None] * d
        n = p / (r * r)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return t0, t1, n


@dataclass
class Convex(Shape):
    """The solid where every plane's n . x <= k (planes as (normal, k))."""
    planes: list = field(default_factory=list)
    bound: Optional[np.ndarray] = None

    def corners(self):
        return self.bound

    def span(self, o, d):
        n_rays = len(o)
        t_in = np.full(n_rays, -INF)
        t_out = np.full(n_rays, INF)
        n_in = np.zeros((n_rays, 3))
        for n, k in self.planes:
            den = float(n @ d)
            num = k - o @ n
            if abs(den) < 1e-12:
                t_in = np.where(num < 0, INF, t_in)
                continue
            t = num / den
            if den < 0:
                later = t > t_in
                t_in = np.where(later, t, t_in)
                n_in = np.where(later[:, None], n, n_in)
            else:
                t_out = np.minimum(t_out, t)
        miss = ~(t_in <= t_out)
        return np.where(miss, INF, t_in), t_out, n_in


def _plane(n, point) -> tuple:
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return n, float(n @ np.asarray(point, float))


def convex_from_points(points: np.ndarray, material: Material, xf=None) -> Convex:
    """The convex hull of a few points (a brute-force hull: fine for the handful a wedge or a blade has)."""
    pts = np.asarray(points, float)
    c = pts.mean(0)
    planes = []
    m = len(pts)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                n = np.cross(pts[j] - pts[i], pts[k] - pts[i])
                ln = np.linalg.norm(n)
                if ln < 1e-9:
                    continue
                n /= ln
                if n @ (c - pts[i]) > 0:
                    n = -n
                side = (pts - pts[i]) @ n
                if (side <= 1e-6).all() and not any(np.allclose(n, p[0], atol=1e-6) for p in planes):
                    planes.append((n, float(n @ pts[i])))
    return Convex(material, xf if xf is not None else np.eye(4), planes=planes, bound=pts)


@dataclass
class Frustum(Shape):
    """A cylinder or cone along local +z from z=0 (radius r0) to z=h (radius r1), capped unless open."""
    r0: float = 1.0
    r1: float = 1.0
    h: float = 1.0
    capped: bool = True

    def corners(self):
        r = max(self.r0, self.r1)
        return np.array([[sx * r, sy * r, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, self.h)])

    def _side(self, o, d):
        b = (self.r1 - self.r0) / self.h
        a = self.r0
        A = d[0] ** 2 + d[1] ** 2 - (b * d[2]) ** 2
        B = 2 * (o[:, 0] * d[0] + o[:, 1] * d[1] - b * d[2] * (a + b * o[:, 2]))
        C = o[:, 0] ** 2 + o[:, 1] ** 2 - (a + b * o[:, 2]) ** 2
        out = []
        if abs(A) < 1e-12:
            t = np.where(np.abs(B) > 1e-12, -C / np.where(np.abs(B) > 1e-12, B, 1), INF)
            out.append(t)
        else:
            disc = B * B - 4 * A * C
            ok = disc >= 0
            sq = np.sqrt(np.where(ok, disc, 0))
            out += [np.where(ok, (-B - sq) / (2 * A), INF), np.where(ok, (-B + sq) / (2 * A), INF)]
        res = []
        for t in out:
            p = o + np.where(np.isfinite(t), t, 0)[:, None] * d
            z = p[:, 2]
            r = a + b * z
            valid = np.isfinite(t) & (z >= 0) & (z <= self.h) & (r >= 0)
            n = np.stack([p[:, 0], p[:, 1], -b * r], 1)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            res.append((np.where(valid, t, np.nan), n))
        return res

    def span(self, o, d):
        cands = self._side(o, d)
        if self.capped:
            for z, r, nz in ((0.0, self.r0, -1.0), (self.h, self.r1, 1.0)):
                if abs(d[2]) < 1e-12:
                    continue
                t = (z - o[:, 2]) / d[2]
                p = o + t[:, None] * d
                inside = p[:, 0] ** 2 + p[:, 1] ** 2 <= r * r
                n = np.zeros((len(o), 3))
                n[:, 2] = nz
                cands.append((np.where(inside, t, np.nan), n))
        n_rays = len(o)
        t_in = np.full(n_rays, INF)
        t_out = np.full(n_rays, -INF)
        n_in = np.zeros((n_rays, 3))
        for t, n in cands:
            front = (n @ d) < 0
            ok = np.isfinite(t)
            enter = ok & front & (t < t_in)
            t_in = np.where(enter, t, t_in)
            n_in = np.where(enter[:, None], n, n_in)
            t_out = np.where(ok & ~front & (t > t_out), t, t_out)
        if not self.capped:  # an open tube: the inside shows where the front is missing
            t_out = np.where(np.isfinite(t_in), np.maximum(t_out, t_in), t_out)
        return t_in, t_out, n_in


@dataclass
class Lathe(Shape):
    """A profile [(r, z), ...] turned round local +z. Closed shapes (ending on the axis, or a loop) look solid;
    normals are smoothed along the profile so a few points make a smooth curve."""
    profile: list = field(default_factory=list)
    convex = False

    def corners(self):
        pr = np.asarray(self.profile, float)
        r = pr[:, 0].max()
        return np.array([[sx * r, sy * r, z] for sx in (-1, 1) for sy in (-1, 1) for z in (pr[:, 1].min(),
                                                                                         pr[:, 1].max())])

    def nearest(self, o, d):
        pr = np.asarray(self.profile, float)
        seg = pr[1:] - pr[:-1]
        # smoothed normals at the profile's points (in the (r, z) plane, pointing away from the axis side)
        sn = np.stack([seg[:, 1], -seg[:, 0]], 1)
        sn /= np.linalg.norm(sn, axis=1, keepdims=True) + 1e-12
        vn = np.zeros_like(pr)
        vn[:-1] += sn
        vn[1:] += sn
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        n_rays = len(o)
        best = np.full(n_rays, INF)
        nb = np.zeros((n_rays, 3))
        for i in range(len(seg)):
            (ra, za), (rb, zb) = pr[i], pr[i + 1]
            if abs(zb - za) < 1e-9:  # a flat ring: a plane z = za between two radii
                if abs(d[2]) < 1e-12:
                    continue
                t = (za - o[:, 2]) / d[2]
                p = o + t[:, None] * d
                rr = np.hypot(p[:, 0], p[:, 1])
                ok = (rr >= min(ra, rb)) & (rr <= max(ra, rb)) & (t < best)
                nz = np.sign(sn[i, 1]) if sn[i, 1] != 0 else 1.0
                best = np.where(ok, t, best)
                nb = np.where(ok[:, None], np.array([0, 0, nz]), nb)
                continue
            lo, hi = (za, zb) if za < zb else (zb, za)
            b = (rb - ra) / (zb - za)
            a = ra - b * za
            A = d[0] ** 2 + d[1] ** 2 - (b * d[2]) ** 2
            B = 2 * (o[:, 0] * d[0] + o[:, 1] * d[1] - b * d[2] * (a + b * o[:, 2]))
            C = o[:, 0] ** 2 + o[:, 1] ** 2 - (a + b * o[:, 2]) ** 2
            if abs(A) < 1e-12:
                roots = [np.where(np.abs(B) > 1e-12, -C / np.where(np.abs(B) > 1e-12, B, 1), INF)]
            else:
                disc = B * B - 4 * A * C
                ok = disc >= 0
                sq = np.sqrt(np.where(ok, disc, 0))
                roots = [np.where(ok, (-B - sq) / (2 * A), INF), np.where(ok, (-B + sq) / (2 * A), INF)]
            for t in roots:
                p = o + np.where(np.isfinite(t), t, 0)[:, None] * d
                z = p[:, 2]
                r = a + b * z
                ok = np.isfinite(t) & (z >= lo) & (z <= hi) & (r >= 0) & (t < best)
                if not ok.any():
                    continue
                f = np.clip((z - za) / (zb - za), 0, 1)[:, None]
                n2 = vn[i] * (1 - f) + vn[i + 1] * f  # (radial, z) smoothed
                rr = np.hypot(p[:, 0], p[:, 1]) + 1e-12
                n = np.stack([p[:, 0] / rr * n2[:, 0], p[:, 1] / rr * n2[:, 0], n2[:, 1]], 1)
                best = np.where(ok, t, best)
                nb = np.where(ok[:, None], n, nb)
        flip = (nb @ d) > 0  # always face the viewer
        nb = np.where(flip[:, None], -nb, nb)
        nb /= np.linalg.norm(nb, axis=1, keepdims=True) + 1e-12
        return best, nb


# --------------------------------------------------------------------------- convenient constructors

def ball(at, r, m: Material, rot=(0, 0, 0)) -> Ellipsoid:
    rr = (r, r, r) if np.isscalar(r) else tuple(r)
    return Ellipsoid(m, _orient(at, rot), radii=rr)


def rod(a, b, r0, m: Material, r1=None, capped=True) -> Frustum:
    """A cylinder or cone from point a to point b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return Frustum(m, axis_frame(a, b), r0=r0, r1=r0 if r1 is None else r1, h=float(np.linalg.norm(b - a)),
                   capped=capped)


def limb(a, b, r0, m: Material, r1=None) -> list:
    """A rod with rounded ends (a capsule, tapering from r0 to r1)."""
    r1 = r0 if r1 is None else r1
    return [rod(a, b, r0, m, r1), ball(a, r0, m), ball(b, r1, m)]


def turned(at, profile, m: Material, rot=(0, 0, 0)) -> Lathe:
    return Lathe(m, _orient(at, rot), profile=list(profile))


def block(lo, hi, m: Material, rot=(0, 0, 0), at=None) -> Convex:
    """A box from its low to its high corner (turned by rot about its centre, or placed at `at`)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    planes = [(np.array(v, float), float(s)) for v, s in (((1, 0, 0), h[0]), ((-1, 0, 0), h[0]), ((0, 1, 0), h[1]),
                                                          ((0, -1, 0), h[1]), ((0, 0, 1), h[2]), ((0, 0, -1), h[2]))]
    bound = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return Convex(m, _orient(c if at is None else at, rot), planes=planes, bound=bound)


def hull(points, m: Material, at=(0, 0, 0), rot=(0, 0, 0)) -> Convex:
    return convex_from_points(np.asarray(points, float), m, _orient(at, rot))


def torus(at, R, r, m: Material, rot=(0, 0, 0), steps=14) -> Lathe:
    a = np.linspace(0, 2 * np.pi, steps + 1)
    return Lathe(m, _orient(at, rot), profile=[(R + r * np.cos(x), r * np.sin(x)) for x in a])


# --------------------------------------------------------------------------- rendering

def _items(root, pose: Pose, heading: float):
    """Every shape and old box, with its world transform."""
    placed = place(root, pose, model_matrix(heading))
    out = []
    for pb in placed:
        if isinstance(pb.box, Shape):
            out.append((pb.box, pb.matrix @ pb.box.xf))
        else:
            out.append((pb.box, pb.matrix))
    return out


def _world_corners(item, M) -> np.ndarray:
    if isinstance(item, Shape):
        c = item.corners()
        return c @ M[:3, :3].T + M[:3, 3]
    return _corners(PlacedBox(item, M))


def fit_camera(root, heading: float, pose: Pose = None, scale: float = 1.5, elevation: float = 30.0,
               pad: int = 4) -> Camera:
    items = _items(root, pose or Pose(), heading)
    pts = np.concatenate([_world_corners(it, M) for it, M in items])
    pts = np.concatenate([pts, pts - (pts[:, 2:3] / SUN[2]) * SUN])
    a = np.radians(elevation)
    x = pts[:, 0] * scale
    y = -(pts[:, 1] * np.sin(a) + pts[:, 2] * np.cos(a)) * scale
    x0, x1 = int(np.floor(x.min())) - pad, int(np.ceil(x.max())) + pad
    y0, y1 = int(np.floor(y.min())) - pad, int(np.ceil(y.max())) + pad
    return Camera(width=x1 - x0, height=y1 - y0, origin=(-x0, -y0), scale=scale, elevation=elevation)


def _box_hits(box: Box, M, O, D, win, depth):
    pb = PlacedBox(box, M)
    o, d, tnear, tfar, axis = _slabs(pb, O[win], D)
    hit = np.nonzero((tfar >= tnear) & (tnear < depth[win]))[0]
    rows, ts, normals, rgb, team = [], [], [], [], []
    if not len(hit):
        return None
    p = o[hit] + tnear[hit, None] * d
    for k in range(3):
        sel = axis[hit] == k
        if not sel.any():
            continue
        face = _FACE[(k, bool(d[k] < 0))]
        tex = box.faces[face]
        u, v = _uv(face, p[sel], box.lo, box.hi)
        th, tw = tex.shape[:2]
        tx = tex[np.clip((v * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1)]
        ok = tx[:, 3] > 0
        local = hit[sel][ok]
        rows.append(win[local])
        ts.append(tnear[local])
        normals.append(np.broadcast_to(M[:3, :3] @ np.array(_NORMAL[face], float), (len(local), 3)))
        rgb.append(tx[ok, :3])
        team.append(tx[ok, 4] > 0)
    if not rows:
        return None
    return (np.concatenate(rows), np.concatenate(ts), np.concatenate(normals), np.concatenate(rgb),
            np.concatenate(team))


def _light(nw: np.ndarray, view: np.ndarray, shine: np.ndarray, gloss: np.ndarray, z: np.ndarray):
    """Cartoon light: wrapped key light, sky/ground fill, a moonlit rim, a highlight. Returns (diffuse rgb factor,
    additive rgb, lightness for player colour)."""
    k = np.clip((nw @ KEY + WRAP) / (1 + WRAP), 0, 1)
    k = k * k * (3 - 2 * k)  # soft but with a clear turn from light to shade
    up = 0.5 + 0.5 * nw[:, 2]
    fill = GROUND_RGB[None] * (1 - up[:, None]) + SKY_RGB[None] * up[:, None]
    occl = 0.78 + 0.22 * np.clip(z / 6.0, 0, 1)  # a little darker near the ground
    diffuse = (fill * 0.62 + KEY_RGB[None] * k[:, None] * 0.78) * occl[:, None]
    facing = np.clip(nw @ view, 0, 1)
    rim = (1 - facing) ** 3 * np.clip(0.4 - nw @ KEY, 0, 1) * 0.9
    h = KEY + view
    h /= np.linalg.norm(h)
    spec = shine * np.clip(nw @ h, 0, 1) ** gloss
    add = rim[:, None] * RIM_RGB[None] + spec[:, None]
    lightness = np.clip(0.18 + 0.85 * k * occl + 0.35 * rim + spec, 0, 1)
    return diffuse, add, lightness


def render(root, heading: float, pose: Pose = None, camera: Camera = None, shadow: bool = True,
           outline: bool = False, samples: int = SAMPLES, ink: bool = True) -> Frame:
    pose = pose or Pose()
    camera = camera or Camera()
    S = samples
    cam = Camera(width=camera.width * S, height=camera.height * S, origin=(camera.origin[0] * S, camera.origin[1] * S),
                 scale=camera.scale * S, elevation=camera.elevation)
    items = _items(root, pose, heading)
    O, D = cam.rays()
    n = len(O)
    depth = np.full(n, INF)
    owner = np.full(n, -1, np.int32)
    normal = np.zeros((n, 3))
    local_p = np.zeros((n, 3))
    local_n = np.zeros((n, 3))
    cutface = np.zeros(n, bool)
    box_rgb = np.zeros((n, 3))
    box_team = np.zeros(n, bool)

    for idx, (it, M) in enumerate(items):
        win = cam.window(_world_corners(it, M))
        if not len(win):
            continue
        if not isinstance(it, Shape):
            res = _box_hits(it, M, O, D, win, depth)
            if res is None:
                continue
            rows, ts, nw, rgb, team = res
            closer = ts < depth[rows]
            rows, ts, nw, rgb, team = rows[closer], ts[closer], nw[closer], rgb[closer], team[closer]
            depth[rows], owner[rows], normal[rows] = ts, idx, nw
            box_rgb[rows], box_team[rows] = rgb, team
            continue
        R, t = M[:3, :3], M[:3, 3]
        o = (O[win] - t) @ R
        d = R.T @ D
        th, nl, cut = it.trace(o, d)
        closer = th < depth[win]
        if not closer.any():
            continue
        rows = win[closer]
        depth[rows] = th[closer]
        owner[rows] = idx
        local_n[rows] = nl[closer]
        local_p[rows] = o[closer] + th[closer, None] * d
        normal[rows] = nl[closer] @ R.T
        cutface[rows] = cut[closer]

    drawn = owner >= 0
    rgb = np.zeros((n, 3))
    light = np.zeros(n)
    kind = np.full(n, TRANSPARENT, np.uint8)
    view = -D / np.linalg.norm(D)
    world_z = (O + np.where(drawn, depth, 0)[:, None] * D)[:, 2]
    for idx in np.unique(owner[drawn]):
        rows = np.nonzero(owner == idx)[0]
        it, M = items[idx]
        if isinstance(it, Shape):
            m = it.material
            nl = local_n[rows]
            look = Look(local_p[rows], nl, normal[rows], np.where(cutface[rows], "cut", "surface"))
            cut_rows = cutface[rows]
            base, glow, team = m.evaluate(look)
            if m.cut is not None and cut_rows.any():
                cl = Look(look.p[cut_rows], nl[cut_rows], look.nw[cut_rows], look.face[cut_rows])
                cb, cg, ct = m.cut.evaluate(cl)
                base, glow, team = base.copy(), glow.copy(), team.copy()
                base[cut_rows], glow[cut_rows], team[cut_rows] = cb, cg, ct
            nw = normal[rows]
            if m.bump is not None:
                bumped = m.bump(look) @ M[:3, :3].T
                nw = np.where(cut_rows[:, None], nw, bumped)
            nw = nw / (np.linalg.norm(nw, axis=1, keepdims=True) + 1e-12)
            shine = np.full(len(rows), m.shine)
            gloss = np.full(len(rows), m.gloss)
        else:
            base, nw = box_rgb[rows], normal[rows]
            glow, team = np.zeros(len(rows)), box_team[rows]
            shine, gloss = np.zeros(len(rows)), np.full(len(rows), 8.0)
        diffuse, add, lit = _light(nw, view, shine, gloss[0] if len(gloss) else 8.0, world_z[rows])
        col = base * diffuse + add
        col = col * (1 - glow[:, None]) + base * glow[:, None]
        rgb[rows] = np.clip(col, 0, 1)
        lum = base @ np.array([0.3, 0.55, 0.15])
        light[rows] = np.clip(lit * (0.55 + 0.6 * lum) * (1 - glow) + glow, 0, 1)
        kind[rows] = np.where(team, PLAYER, SOLID)

    if pose.tint is not None:
        r_, g_, b_, s_ = pose.tint
        rgb = np.clip(rgb * (1 - s_) + np.array([r_, g_, b_]) * s_ * rgb.mean(axis=1, keepdims=True) * 2, 0, 1)

    H, W = cam.height, cam.width
    # ink: where depth jumps between neighbouring samples, the nearer side gets a dark line
    if ink:
        dep = np.where(drawn, depth, INF).reshape(H, W)
        inked = np.zeros((H, W), bool)
        thr = INK_DEPTH
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nb = np.full((H, W), INF)
            ys = slice(max(0, dy), H + min(0, dy))
            yd = slice(max(0, -dy), H + min(0, -dy))
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            nb[yd, xd] = dep[ys, xs]
            with np.errstate(invalid="ignore"):
                inked |= np.isfinite(dep) & (nb - dep > thr)  # the neighbour is far behind (or empty)
        own = owner.reshape(H, W)
        inks = np.array([getattr(it, "ink", True) for it, _ in items] + [False])
        inked &= inks[np.where(own >= 0, own, len(items))]
        for _ in range(S // 2):  # about a screen pixel wide, on the near side of the edge
            grow = inked.copy()
            grow[1:, :] |= inked[:-1, :]
            grow[:-1, :] |= inked[1:, :]
            grow[:, 1:] |= inked[:, :-1]
            grow[:, :-1] |= inked[:, 1:]
            inked = grow & np.isfinite(dep) & inks[np.where(own >= 0, own, len(items))]
        flat = inked.ravel()
        rgb[flat] = rgb[flat] * 0.25 + INK * 0.75
        light[flat] *= 0.3

    shadowed = np.zeros(n, bool)
    if shadow:
        ground = O + (-O[:, 2] / D[2])[:, None] * D
        todo = ~drawn
        for it, M in items:
            if isinstance(it, Shape) and not getattr(it, "casts", True):
                continue
            c = _world_corners(it, M)
            foot = c - (c[:, 2:3] / SUN[2]) * SUN
            win = cam.window(foot)
            if not len(win):
                continue
            win = win[todo[win] & ~shadowed[win]]
            if not len(win):
                continue
            g = ground[win] + SUN * 0.01
            if isinstance(it, Shape):
                R, t = M[:3, :3], M[:3, 3]
                th, _, _ = it.trace((g - t) @ R, R.T @ SUN)
                shadowed[win] |= np.isfinite(th) & (th > 0)
            else:
                _, _, tnear, tfar, _ = _slabs(PlacedBox(it, M), g, SUN)
                shadowed[win] |= (tfar >= tnear) & (tfar > 0)

    # down to screen pixels: a pixel is drawn when most of its samples are
    h, w = camera.height, camera.width

    def blocks(a):
        extra = a.shape[2:]
        return a.reshape(h, S, w, S, *extra).swapaxes(1, 2).reshape(h, w, S * S, *extra)

    kd = blocks(kind.reshape(H, W))
    body = (kd == SOLID) | (kd == PLAYER)
    cover = body.mean(2)
    pc = (kd == PLAYER).sum(2)
    sol = (kd == SOLID).sum(2)
    out_kind = np.full((h, w), TRANSPARENT, np.uint8)
    drawn_px = cover >= 0.5
    out_kind[drawn_px & (pc > sol)] = PLAYER
    out_kind[drawn_px & (pc <= sol)] = SOLID
    shade_px = blocks(shadowed.reshape(H, W)).mean(2) >= 0.5
    out_kind[~drawn_px & shade_px] = SHADOW
    rb = blocks(rgb.reshape(H, W, 3))
    wsol = (kd == SOLID)[..., None]
    out_rgb = (rb * wsol).sum(2) / np.maximum(wsol.sum(2), 1)
    # a pixel mostly in player colour still averages its plain samples' colour, in case it is drawn plain
    lb = blocks(light.reshape(H, W))
    wpc = kd == PLAYER
    out_light = (lb * wpc).sum(2) / np.maximum(wpc.sum(2), 1)
    if outline:
        bodyp = (out_kind == SOLID) | (out_kind == PLAYER)
        edge = np.zeros_like(bodyp)
        edge[1:, :] |= bodyp[:-1, :]
        edge[:-1, :] |= bodyp[1:, :]
        edge[:, 1:] |= bodyp[:, :-1]
        edge[:, :-1] |= bodyp[:, 1:]
        out_kind[edge & ~bodyp] = OUTLINE
    return Frame(out_kind, np.clip(out_rgb, 0, 1), np.clip(out_light, 0, 1),
                 (int(round(camera.origin[0])), int(round(camera.origin[1]))))
