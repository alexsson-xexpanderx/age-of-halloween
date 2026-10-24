"""Halloween night pictures for the menus: a night sky, the moon, misty hills, things rendered from the game's own
models (sculpt.py), and the "AGE OF HALLOWEEN" title. All RGB(A) 0..1 numpy pictures, as menu.py and loadscreen.py
draw with.

The title is drawn with a font of the player's own Windows (Georgia Bold, else Times New Roman Bold, else Pillow's
own): letters glowing orange from yellow at the top to blood red at the bottom, a dark outline, a green glow round
them and drops running off their feet.
"""
from __future__ import annotations

import math
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np

FONTS = ("C:/Windows/Fonts/georgiab.ttf", "C:/Windows/Fonts/timesbd.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")


def _font(size: int):
    from PIL import ImageFont
    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    from PIL import Image, ImageFilter
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.float64) / 255


def _text_mask(text: str, size: int, stroke: int = 0) -> np.ndarray:
    from PIL import Image, ImageDraw
    font = _font(size)
    d = ImageDraw.Draw(Image.new("L", (10, 10)))
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
    im = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4))
    ImageDraw.Draw(im).text((-x0 + 2, -y0 + 2), text, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    return np.asarray(im).astype(np.float64) / 255


@lru_cache(maxsize=None)
def title(width: int, height: int = None) -> np.ndarray:
    """The "AGE OF HALLOWEEN" title, `width` pixels wide (and `height` tall: its letters drawn taller, thin and
    spooky, where it has the room): RGBA 0..1."""
    big = 200
    rows = []
    for text, k in (("AGE OF", 0.5), ("HALLOWEEN", 1.0)):
        size = int(big * k)
        fill = _text_mask(text, size, 0)
        edge = _text_mask(text, size, max(2, size // 22))
        h = max(fill.shape[0], edge.shape[0])
        w = max(fill.shape[1], edge.shape[1])
        f2 = np.zeros((h, w))
        e2 = np.zeros((h, w))
        f2[(h - fill.shape[0]) // 2:(h - fill.shape[0]) // 2 + fill.shape[0],
           (w - fill.shape[1]) // 2:(w - fill.shape[1]) // 2 + fill.shape[1]] = fill
        e2[(h - edge.shape[0]) // 2:(h - edge.shape[0]) // 2 + edge.shape[0],
           (w - edge.shape[1]) // 2:(w - edge.shape[1]) // 2 + edge.shape[1]] = edge
        rows.append((f2, e2))
    w = max(f.shape[1] for f, _ in rows)
    gap = 6
    pad = 28
    H = sum(f.shape[0] for f, _ in rows) + gap + 2 * pad + 40
    W = w + 2 * pad
    fill = np.zeros((H, W))
    edge = np.zeros((H, W))
    y = pad
    for f, e in rows:
        x = (W - f.shape[1]) // 2
        fill[y:y + f.shape[0], x:x + f.shape[1]] = np.maximum(fill[y:y + f.shape[0], x:x + f.shape[1]], f)
        edge[y:y + e.shape[0], x:x + e.shape[1]] = np.maximum(edge[y:y + e.shape[0], x:x + e.shape[1]], e)
        y += f.shape[0] + gap
    # drops running off the letters' feet
    rng = np.random.default_rng(zlib.crc32(b"drips"))
    solid = fill > 0.5
    feet = solid & ~np.vstack([solid[1:], np.zeros((1, W), bool)])
    ys, xs = np.nonzero(feet)
    for i in rng.choice(len(ys), size=min(28, len(ys)), replace=False):
        y0, x0 = ys[i], xs[i]
        length = int(rng.uniform(8, 40))
        r = rng.uniform(2.5, 5.0)
        yy, xx = np.mgrid[0:H, 0:W]
        stem = (np.abs(xx - x0) <= r * (1 - 0.4 * np.clip((yy - y0) / length, 0, 1))) & (yy >= y0 - 2) & (yy <= y0 + length)
        drop = np.hypot(xx - x0, yy - (y0 + length)) <= r * 1.25
        fill = np.maximum(fill, (stem | drop).astype(float))
        edge = np.maximum(edge, _grow(stem | drop, 3).astype(float))
    yy = np.linspace(0, 1, H)[:, None]
    top, mid, low = np.array([1.0, 0.86, 0.35]), np.array([1.0, 0.5, 0.08]), np.array([0.62, 0.08, 0.04])
    grad = np.where(yy < 0.55, top + (mid - top) * (yy / 0.55)[..., None],
                    mid + (low - mid) * ((yy - 0.55) / 0.45)[..., None])[:, 0, :]
    rgb = np.broadcast_to(grad[:, None, :], (H, W, 3)).copy()
    shine = np.clip(_blur(fill, 2) - _blur(fill, 6), 0, 1)[..., None]  # a bevel of light on the letters' edges
    rgb = np.clip(rgb + shine * 0.6, 0, 1)
    outline = np.array([0.08, 0.03, 0.08])
    glow = _blur(edge, 10) * 0.85
    out = np.zeros((H, W, 4))
    out[..., :3] = np.array([0.4, 1.0, 0.35])
    out[..., 3] = glow
    a = edge[..., None]
    out[..., :3] = out[..., :3] * (1 - a) + outline * a
    out[..., 3] = np.maximum(out[..., 3], edge)
    f = fill[..., None]
    out[..., :3] = out[..., :3] * (1 - f) + rgb * f
    out[..., 3] = np.maximum(out[..., 3], fill)
    from PIL import Image
    im = Image.fromarray((out * 255).astype(np.uint8), "RGBA")
    im = im.resize((width, height or max(1, int(H * width / W))), Image.LANCZOS)
    return np.asarray(im).astype(np.float64) / 255


def _grow(mask, r):
    m = mask.copy()
    for _ in range(r):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


def over(img: np.ndarray, rgba: np.ndarray, x: int, y: int, masks=None, name=None, cut: float = 0.5) -> None:
    """Paste RGBA over img at (x, y) (clipped); its solid pixels join masks[name]."""
    H, W = img.shape[:2]
    h, w = rgba.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4]
    img[y0:y1, x0:x1] = src[..., :3] * a + img[y0:y1, x0:x1] * (1 - a)
    if masks is not None and name:
        m = masks.setdefault(name, np.zeros((H, W), bool))
        m[y0:y1, x0:x1] |= src[..., 3] > cut


def model(shapes, heading: float = -90.0, width: float = None, height: float = None, elevation: float = 15.0
          ) -> np.ndarray:
    """Shapes (or a geometry.Part) rendered RGBA 0..1 and cropped to them, `width` or `height` pixels big. At the
    default heading a model's +y side faces the viewer."""
    from . import sculpt as S
    from .geometry import Part
    part = shapes if isinstance(shapes, Part) else Part("model", boxes=list(shapes))

    def draw(scale):
        cam = S.fit_camera(part, heading, scale=scale, pad=3, elevation=elevation)
        rgba = S.render(part, heading, camera=cam, shadow=False).to_rgba(1).astype(np.float64) / 255
        ys, xs = np.nonzero(rgba[..., 3] > 0)
        return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    probe = draw(1.0)
    k = width / probe.shape[1] if width else height / probe.shape[0]
    return draw(k)


def sky(img: np.ndarray, top=(0.05, 0.03, 0.10), low=(0.32, 0.16, 0.30), stars: int = 160, seed: int = 1,
        horizon: float = None) -> None:
    """A night sky, deep purple to a dusky glow at the horizon (the picture's foot, or the height `horizon`), with
    stars."""
    H, W = img.shape[:2]
    yy = np.clip(np.arange(H) / (horizon or H), 0, 1)[:, None, None] ** 1.3
    img[:] = np.array(top) * (1 - yy) + np.array(low) * yy
    rng = np.random.default_rng(seed)
    for _ in range(stars):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, int(H * 0.7)))
        b = rng.uniform(0.4, 1.0)
        img[y, x] = np.maximum(img[y, x], b)
        if b > 0.9 and 0 < x < W - 1 and 0 < y < H - 1:
            img[y - 1:y + 2, x] = np.maximum(img[y - 1:y + 2, x], b * 0.5)
            img[y, x - 1:x + 2] = np.maximum(img[y, x - 1:x + 2], b * 0.5)


def moon(img, cx, cy, r, colour=(1.0, 0.92, 0.72), halo=True, seas: float = 0.18) -> None:
    """A full moon with its grey seas (`seas` dark) and a soft halo."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(xx - cx, yy - cy) / r
    if halo:
        glow = np.clip(1.6 - d, 0, 1) ** 2 * 0.45
        img += (np.array(colour) - img) * glow[..., None] * 0.6
    disc = d <= 1.0
    rng = np.random.default_rng(3)
    shade = np.ones((H, W))
    for _ in range(9):
        sx, sy = cx + rng.uniform(-0.6, 0.6) * r, cy + rng.uniform(-0.6, 0.6) * r
        sr = rng.uniform(0.12, 0.35) * r
        shade -= np.clip(1 - np.hypot(xx - sx, yy - sy) / sr, 0, 1) * seas
    limb = np.sqrt(np.clip(1 - d ** 2, 0, 1)) * 0.15 + 0.85
    img[disc] = (np.array(colour)[None] * (shade * limb)[disc][:, None])


def hills(img, ys, colour, seed=0, rough=0.4) -> np.ndarray:
    """A smooth line of hills across the picture (heights `ys` spread over its width), filled with `colour` below;
    returns the ground's height at each column."""
    H, W = img.shape[:2]
    xs = np.linspace(0, W - 1, len(ys))
    line = np.interp(np.arange(W), xs, ys)
    rng = np.random.default_rng(seed)
    noise = np.convolve(rng.normal(0, 1, W + 40), np.ones(20) / 20, "same")[20:W + 20]
    line = line + noise * 6 * rough
    yy = np.arange(H)[:, None]
    below = yy >= line[None, :]
    img[below] = np.array(colour)
    return line


def fog(img, y0, y1, colour=(0.55, 0.62, 0.66), strength=0.45, seed=2) -> None:
    """Bands of mist between heights y0 and y1."""
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    band = np.clip(1 - np.abs((yy - (y0 + y1) / 2) / max(1, (y1 - y0) / 2)), 0, 1)
    wisp = 0.5 + 0.5 * np.sin(xx / 47.0 + np.sin(yy / 13.0) * 2 + rng.uniform(0, 6)) * np.sin(xx / 83.0 + 1.3)
    a = band * wisp * strength
    img += (np.array(colour) - img) * a[..., None]


# --------------------------------------------------------------------------- painting on a flat picture


def field(h: int, w: int, scale: float, seed: int = 0, octaves: int = 3, x0: float = 0, y0: float = 0,
          stretch=(1.0, 1.0)) -> np.ndarray:
    """Smooth noise 0..1 over an h x w area (paint.fbm in the picture's plane), `stretch` (x, y) pulling it out."""
    from . import paint as P
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.stack([(xx + x0) / stretch[0], (yy + y0) / stretch[1], np.zeros_like(xx)], -1).reshape(-1, 3)
    return P.fbm(p, scale, seed, octaves).reshape(h, w)


def shape(h: int, w: int, draw, ss: int = 4) -> np.ndarray:
    """A smooth-edged mask 0..1: draw(d, k) draws on a PIL picture k times bigger."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (w * ss, h * ss))
    draw(ImageDraw.Draw(im), ss)
    return np.asarray(im.resize((w, h), Image.BOX)).astype(np.float64) / 255


def lay(img: np.ndarray, alpha: np.ndarray, colour, box=None) -> None:
    """Paint `colour` (an RGB or a picture) through `alpha` (0..1) on img, or on its box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box or (0, 0, img.shape[1], img.shape[0])
    a = np.clip(alpha, 0, 1)[..., None]
    region = img[y0:y1, x0:x1]
    region[...] = region * (1 - a) + np.asarray(colour, float) * a


def glow(img, cx, cy, r, colour, strength=0.6) -> None:
    """Light falling round a lamp: added light, fading with distance."""
    H, W = img.shape[:2]
    x0, x1, y0, y1 = max(0, int(cx - r)), min(W, int(cx + r) + 1), max(0, int(cy - r)), min(H, int(cy + r) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1) ** 2 * strength
    img[y0:y1, x0:x1] = 1 - (1 - img[y0:y1, x0:x1]) * (1 - d[..., None] * np.asarray(colour, float))


def stones(h: int, w: int, base, dark, bw: float = 30, bh: float = 16, seed: int = 0, moss: float = 0.0,
           mortar=None) -> np.ndarray:
    """Old stone blocks laid in rows, each its own shade, worn and grimy, mortar between them, moss low down."""
    from . import paint as P
    rng = np.random.default_rng(seed)
    ids = np.zeros((h, w), np.int32)
    tone = [0.0]
    y, k = 0, 0
    while y < h:
        rh = max(4, int(round(bh * rng.uniform(0.8, 1.2))))
        x = -int(rng.uniform(0, bw))
        while x < w:
            bw_ = max(5, int(round(bw * rng.uniform(0.65, 1.35))))
            k += 1
            ids[y:y + rh, max(0, x):max(0, x + bw_)] = k
            tone.append(rng.uniform(-0.16, 0.16))
            x += bw_
        y += rh
    right = np.zeros((h, w), bool)
    right[:, :-1] = ids[:, :-1] != ids[:, 1:]
    below = np.zeros((h, w), bool)
    below[:-1] = ids[:-1] != ids[1:]
    joint = right | below
    joint[:, 1:] |= right[:, :-1]
    joint[1:] |= below[:-1]
    lit = np.zeros((h, w), bool)  # each block's top edge catches the light, its bottom is in shade
    lit[1:] = joint[:-1] & ~joint[1:]
    shade = np.zeros((h, w), bool)
    shade[:-1] = joint[1:] & ~joint[:-1]
    grime = field(h, w, 14, seed + 1)
    t = np.asarray(tone)[ids] + (grime - 0.5) * 0.5 + (field(h, w, 3, seed + 2, 2) - 0.5) * 0.25
    out = P.mix(dark, base, np.clip(0.55 + t, 0, 1))
    out[lit] = out[lit] * 1.25
    out[shade] = out[shade] * 0.6
    out[joint] = np.asarray(mortar if mortar is not None else np.asarray(dark, float) * 0.45, float)
    if moss:
        g = field(h, w, 9, seed + 3) + np.linspace(-0.35, 0.25, h)[:, None] * moss * 2
        m = np.clip((g - 0.62) * 4, 0, 1) * moss
        out = P.mix(out, np.array([0.16, 0.24, 0.12]) * (0.7 + grime[..., None] * 0.6), m)
    return np.clip(out, 0, 1)


def boards(h: int, w: int, base, dark, bw: float = 13, seed: int = 0, across: bool = False, nails: bool = None
           ) -> np.ndarray:
    """Weathered planks side by side (upright, or `across`), with grain, gaps, knots and nail heads (at the ends of
    short planks, or as asked)."""
    from . import paint as P
    if across:
        return np.transpose(boards(w, h, base, dark, bw, seed, nails=nails), (1, 0, 2))
    nails = h <= 160 if nails is None else nails
    rng = np.random.default_rng(seed)
    edges = [0]
    while edges[-1] < w:
        edges.append(edges[-1] + max(5, int(round(bw * rng.uniform(0.8, 1.25)))))
    ids = np.searchsorted(np.asarray(edges[1:]), np.arange(w), side="right")
    tone = rng.uniform(-0.18, 0.18, len(edges))[ids][None, :]
    grain = field(h, w, 2.2, seed + 1, 3, stretch=(1.0, 14.0))
    worn = field(h, w, 18, seed + 2)
    t = 0.5 + tone + (grain - 0.5) * 0.9 + (worn - 0.5) * 0.4
    out = P.mix(dark, base, np.clip(t, 0, 1))
    gap = np.zeros(w, bool)
    inner = [e for e in edges[1:-1] if e < w]
    gap[inner] = True
    out[:, gap] = np.asarray(dark, float) * 0.35
    lit = np.roll(gap, 1) & ~gap
    out[:, lit] *= 1.18
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(max(1, w * h // 6000)):  # knots
        kx, ky = rng.uniform(0, w), rng.uniform(0, h)
        d = np.hypot((xx - kx) / 1.6, (yy - ky) / 3.2)
        out[d < 1.0] = np.asarray(dark, float) * 0.6
        out[(d >= 1.0) & (d < 1.8)] *= 0.82
    for e0, e1 in zip(edges, edges[1:]) if nails else ():  # nail heads near the ends
        cx = int((e0 + min(e1, w)) / 2)
        for ny in (3, h - 4):
            if 1 <= ny < h and 1 <= cx < w:
                out[ny - 1:ny + 1, cx - 1:cx + 1] = (0.10, 0.09, 0.10)
    return np.clip(out, 0, 1)


def silhouette_tree(d, x, y, height, k, seed, width=1.0) -> None:
    """A bare, crooked tree drawn as a silhouette on a PIL picture (k times bigger), its foot at (x, y)."""
    rng = np.random.default_rng(seed)

    def branch(x0, y0, angle, length, thick, depth):
        x1 = x0 + math.cos(angle) * length
        y1 = y0 - math.sin(angle) * length
        d.line([(x0 * k, y0 * k), (x1 * k, y1 * k)], fill=255, width=max(1, int(thick * k)))
        d.ellipse([(x1 - thick / 2) * k, (y1 - thick / 2) * k, (x1 + thick / 2) * k, (y1 + thick / 2) * k], fill=255)
        if depth == 0 or length < 3:
            return
        for _ in range(2 if depth < 4 else 3):
            branch(x1, y1, angle + rng.uniform(-0.75, 0.75), length * rng.uniform(0.55, 0.78), thick * 0.62,
                   depth - 1)

    trunk = height * 0.36
    x1, y1 = x + rng.uniform(-0.1, 0.1) * height, y - trunk
    d.polygon([((x - height * 0.05 * width) * k, y * k), ((x + height * 0.05 * width) * k, y * k),
               ((x1 + height * 0.02) * k, y1 * k), ((x1 - height * 0.02) * k, y1 * k)], fill=255)
    for a in (math.pi / 2 + 0.5, math.pi / 2 - 0.45, math.pi / 2 + 0.05):
        branch(x1, y1, a + rng.uniform(-0.2, 0.2), height * rng.uniform(0.26, 0.36), height * 0.045, 4)


def bat(d, x, y, size, k, flap=0.0) -> None:
    """A bat's silhouette, wings spread, centred at (x, y)."""
    s = size
    up = -0.35 * flap
    pts = [(-1.0, -0.15 + up), (-0.75, 0.05), (-0.55, -0.02), (-0.4, 0.12), (-0.2, 0.05), (-0.08, 0.15),
           (0, 0.05), (0.08, 0.15), (0.2, 0.05), (0.4, 0.12), (0.55, -0.02), (0.75, 0.05), (1.0, -0.15 + up),
           (0.6, -0.3), (0.25, -0.18), (0.1, -0.28), (0.06, -0.38), (0, -0.25), (-0.06, -0.38), (-0.1, -0.28),
           (-0.25, -0.18), (-0.6, -0.3)]
    d.polygon([((x + px * s) * k, (y + py * s) * k) for px, py in pts], fill=255)


def web(d, cx, cy, r, k, a0: float, a1: float, spokes: int = 6, rings: int = 5) -> None:
    """A spider's web in a corner: spokes from (cx, cy) between angles a0 and a1, sagging threads across them."""
    angles = np.linspace(a0, a1, spokes)
    for a in angles:
        d.line([(cx * k, cy * k), ((cx + math.cos(a) * r) * k, (cy + math.sin(a) * r) * k)], fill=255, width=k)
    for j in range(1, rings + 1):
        rr = r * j / (rings + 0.4)
        for a, b in zip(angles, angles[1:]):
            pts = []
            for t in np.linspace(0, 1, 8):
                ang = a + (b - a) * t
                sag = rr * (1 - 0.12 * math.sin(math.pi * t))
                pts.append(((cx + math.cos(ang) * sag) * k, (cy + math.sin(ang) * sag) * k))
            d.line(pts, fill=255, width=k)
