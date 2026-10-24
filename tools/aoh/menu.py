"""The main menu, Halloween style: a haunted house and a stone tower under a full moon.

The Conquerors' main menu (interfac.drs 50189) is one 800x600 picture (frame 0) and, for every button, pictures
drawn over it at the button's place: the object cut out, with a yellow glow (the mouse is on it), with a white glow
(pressed), and plain. The game writes the button names above the objects, and the Single Player menu (title,
buttons, description) on the right, over the picture.

`scene()` draws the new picture: a Halloween thing in each button's place (models drawn like the units, by
sculpt.py: trinkets.py), and a board of dark planks wherever the game writes a name (`SIGNS`, measured on a
screenshot of the game); on the left a churchyard at night, a haunted house and a round tower under a full moon with
the "AGE OF HALLOWEEN" title over it (nightscape.py), on the right a dark stone wall with cobwebs. `pictures()`
cuts every button picture out of it at the same place, the thing darker while the mouse is on it (`DARKER`, as the
player asked for Age of Minecraft; no glow); every picture keeps its size and hotspot. The places were found by
matching the original button pictures against the original picture; the build leaves a menu whose picture sizes
differ from these alone.

The menu's palette (50589) was made for the original night in the snow, so the menu gets its own (`palette`), and
its screen files (`settings`) have the game draw the Single Player menu's buttons dark purple with white names,
pumpkin orange under the mouse.
"""
from __future__ import annotations

import math
import zlib
from functools import lru_cache
from typing import Optional

import numpy as np

from . import nightscape as N
from . import paint as P
from . import slp
from . import trinkets as T
from .palette import Quantiser
from .textures import parse

W, H = 800, 600
MENU = 50189  # interfac.drs: the Conquerors main menu, drawn with its own palette
PALETTE = 50589
# button -> (its pictures, top left of the cut-out one, top left of the others), on the 800x600 picture
BUTTONS = {
    "single": ((10, 11, 12, 13), (313, 14), (309, 12)),
    "multi": ((14, 15, 16, 17), (267, 221), (263, 217)),
    "zone": ((18, 19, 20, 21), (275, 371), (271, 367)),
    "learn": ((22, 23, 24, 25), (0, 9), (0, 6)),
    "map": ((26, 27, 28, 29), (200, 277), (197, 273)),
    "history": ((30, 31, 32, 33), (107, 168), (103, 164)),
    "options": ((34, 35, 36, 37), (104, 352), (101, 347)),
    "exit": ((46, 47, 48), (7, 540), (5, 535)),
    "logo": ((49,), (107, 36), None),
    "banner": ((50, 51, 52), None, (0, 0)),
}
SIZES = {0: (800, 600), 10: (109, 184), 11: (120, 189), 14: (88, 124), 15: (97, 131), 18: (61, 56), 19: (67, 64),
         22: (121, 179), 23: (125, 187), 26: (63, 60), 27: (70, 67), 30: (94, 89), 31: (102, 97), 34: (87, 91),
         35: (95, 100), 46: (154, 60), 47: (153, 65), 49: (211, 111), 50: (427, 170)}
# where the game writes each button's name (measured on a screenshot of the game), with a board of dark planks
# behind it: all of the name on it, with room around
SIGNS = {"learn": (8, 9, 116, 31), "single": (313, 13, 423, 37), "history": (121, 166, 188, 188),
         "multi": (263, 218, 360, 241), "map": (181, 273, 276, 296), "options": (112, 348, 182, 371),
         "zone": (279, 367, 327, 388)}
TITLE_SIGN = (453, 3, 775, 53)
DESCRIPTION = (380, 490, 790, 597)
# the Single Player menu's six buttons (measured on photos of the game): the game draws them, its way
SUBMENU = [(462, y, 759, y + 39) for y in (90, 155, 221, 286, 351, 417)]
# the buttons darker while the mouse is on them and when pressed, instead of a glow (the player asked); only the
# thing, not the button its name is on
DARKER = {name: (0.72, 0.55) for name in ("single", "multi", "zone", "learn", "map", "history", "options", "exit")}
SKY = ("#0a0614", "#4a2244")  # deep purple overhead, a dusky glow low down
BUTTON = {"fill": "#4a3428", "light": "#8a6a4e", "shade": "#1a100a", "edge": "#000000"}  # the name boards' planks
TOOLTIP = ("#120a16", "#e07a1a", "#6a2408")  # near black, an orange edge fading to rust
# the menu's screen file: the game fills the Single Player menu's buttons dark purple, with light and dark edges
# (outside in, top and left then bottom and right) and a black outline; names in white, the one the mouse is on in
# pumpkin orange
SCREEN = {"background_color": "#2e2236", "bevel_colors": ("#000000", "#7a5a8a", "#5a4068", "#24182c", "#160e1c",
                                                          "#000000"),
          "text_color1": "255 255 255", "text_color2": "0 0 0", "focus_color1": "255 176 64",
          "focus_color2": "0 0 0"}


class Nearest:
    """Nearest-colour quantiser over a whole palette (the menu's own: no player colours to keep free)."""

    def __init__(self, palette: np.ndarray):
        from .palette import _lab
        self.palette = np.asarray(palette)
        self._lab = _lab
        self.lab = _lab(self.palette[:, :3].astype(np.float64))

    def indices(self, rgb: np.ndarray) -> np.ndarray:
        keys = (rgb[:, 0].astype(np.int64) << 16) | (rgb[:, 1].astype(np.int64) << 8) | rgb[:, 2].astype(np.int64)
        uniq, inverse = np.unique(keys, return_inverse=True)
        cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], -1).astype(np.float64)
        best = np.empty(len(uniq), np.int64)
        for a in range(0, len(uniq), 4096):
            d = ((self._lab(cols[a:a + 4096])[:, None] - self.lab[None]) ** 2).sum(-1)
            best[a:a + 4096] = d.argmin(1)
        return best[inverse]


def palette(original: np.ndarray, extra: Optional[np.ndarray] = None) -> np.ndarray:
    """The menu's own palette: the menu's first one has no greens and few sky blues (it was made for a night in the
    snow). Made from the new picture and `extra` colours (other pictures shown with it), keeping the twenty
    Windows colours at 0-9 and 246-255 where they are."""
    from .loadscreen import palette as median_cut
    img, _ = scene()
    swatches = [parse(c)[:3] for c in (BUTTON["fill"], BUTTON["light"], BUTTON["shade"], SCREEN["background_color"],
                                       *SCREEN["bevel_colors"], "#ffffff", "#ffb040", *TOOLTIP)]
    rows = [img.reshape(-1, 3), np.repeat(np.array(swatches), 400, 0)]
    if extra is not None:
        rows.append(np.asarray(extra, float).reshape(-1, 3))
    return median_cut(original, np.concatenate(rows)[:, None, :])


def settings(palette: np.ndarray) -> dict[str, list[str]]:
    """The menu screen file's settings for these colours, the palette ones as indices into `palette` (never the
    Windows colours at 0-9 and 246-255: 0 would mean no fill)."""
    pal = np.asarray(palette, float)[:, :3]
    free = np.arange(10, 246)

    def index(colour: str) -> str:
        return str(int(free[np.abs(pal[free] - parse(colour)[:3] * 255).sum(1).argmin()]))

    out = {k: v.split() for k, v in SCREEN.items() if isinstance(v, str) and not v.startswith("#")}
    out["background_color"] = [index(SCREEN["background_color"])]
    out["bevel_colors"] = [index(c) for c in SCREEN["bevel_colors"]]
    return out


# --------------------------------------------------------------------------- the picture

LEFT = 430  # the night scene; right of it, where the game opens the Single Player menu, a dark stone wall
MOON = (212, 92, 64)  # behind the title
ORANGE = P.hexrgb("#ff9a2a")
GREEN = P.hexrgb("#7cff6a")
INK = P.hexrgb("#0c080e")


def _scene_far(img) -> None:
    """The sky, the moon, thin clouds across it, far hills with a church and bare trees, and mist."""
    N.sky(img, top=P.hexrgb(SKY[0]), low=P.hexrgb(SKY[1]), stars=240, seed=11, horizon=330)
    N.moon(img, *MOON)
    for y, x0, x1, a in ((118, 120, 300, 0.55), (131, 160, 270, 0.45), (52, 230, 320, 0.35)):  # clouds over the moon
        m = N.shape(H, W, lambda d, k, y=y, x0=x0, x1=x1: d.ellipse([x0 * k, (y - 5) * k, x1 * k, (y + 5) * k], fill=255))
        N.lay(img, N._blur(m, 3) * a, P.hexrgb("#2a1a34"))
    far = N.hills(img, [300, 288, 296, 312, 300, 290, 298, 314, 306, 296], P.hexrgb("#24162e"), seed=1)

    def church(d, k):
        d.rectangle([236 * k, 268 * k, 266 * k, int(far[250]) * k], fill=255)
        d.polygon([(232 * k, 270 * k), (251 * k, 252 * k), (270 * k, 270 * k)], fill=255)
        d.rectangle([243 * k, 230 * k, 253 * k, 262 * k], fill=255)
        d.polygon([(241 * k, 232 * k), (248 * k, 204 * k), (255 * k, 232 * k)], fill=255)
        d.rectangle([247 * k, 196 * k, 249 * k, 206 * k], fill=255)
        d.rectangle([244 * k, 199 * k, 252 * k, 201 * k], fill=255)
        N.silhouette_tree(d, 214, int(far[214]) + 2, 58, k, 3)
        N.silhouette_tree(d, 286, int(far[286]) + 2, 46, k, 4)
        N.silhouette_tree(d, 70, 302, 130, k, 5, width=1.4)  # behind the house

    N.lay(img, N.shape(H, W, church), P.hexrgb("#140c1a"))
    for x, y in ((241, 278), (259, 278)):  # the church's lit windows
        img[y:y + 5, x:x + 2] = P.hexrgb("#ffb050")
    N.fog(img[:, :LEFT], 280, 350, colour=P.hexrgb("#7a7a9a"), strength=0.4, seed=4)
    N.hills(img, [372, 362, 356, 366, 376, 368, 360, 366], P.hexrgb("#120a18"), seed=2)
    N.fog(img[:, :LEFT], 380, 470, colour=P.hexrgb("#5a6078"), strength=0.4, seed=5)

    def bats(d, k):
        for x, y, s, f in ((262, 168, 8, 0.4), (282, 150, 6, -0.3), (236, 194, 7, 0.8), (24, 212, 6, 0.2),
                           (150, 152, 5, -0.5)):
            N.bat(d, x, y, s, k, f)

    N.lay(img, N.shape(H, W, bats), INK)


def _house(img) -> None:
    """The haunted house on the left: an iron railing round its roof, a stone ledge, old boards with a lit window,
    a black cat on its sill, a stone ground floor with an arched door and a lamp."""
    wall = N.boards(74, 196, P.hexrgb("#4a3e46"), P.hexrgb("#16111a"), bw=12, seed=21)
    wall *= np.linspace(0.75, 1.0, 196)[None, :, None]
    img[266:340, 0:196] = wall
    N.glow(img, 72, 302, 78, ORANGE, 0.42)
    img[258:266, 0:196] = N.stones(8, 196, P.hexrgb("#6a6672"), P.hexrgb("#2a2830"), bw=24, bh=8, seed=22)
    img[256:258, 0:196] = INK
    img[332:348, 0:196] = N.boards(16, 196, P.hexrgb("#3a2a22"), P.hexrgb("#120c0a"), bw=10, seed=23, across=True)
    low = N.stones(122, 196, P.hexrgb("#55525e"), P.hexrgb("#1a1820"), bw=28, bh=15, seed=24, moss=0.7)
    img[348:470, 0:196] = low * np.linspace(0.7, 1.0, 196)[None, :, None]
    for a in (0, 184):  # the corner posts
        img[258:470, a:a + 12] = N.boards(212, 12, P.hexrgb("#3a2a22"), P.hexrgb("#120c0a"), bw=12, seed=25 + a)
        img[258:470, a + 11 if a == 0 else a] = INK
    # the window: warm light, a cross of glazing bars, a broken pane, the cat, crooked shutters
    yy, xx = np.mgrid[280:324, 40:104]
    d = np.clip(np.hypot((xx - 72) / 40, (yy - 304) / 26), 0, 1)
    img[280:324, 40:104] = P.mix(P.hexrgb("#ffe6a0"), P.hexrgb("#c8460c"), d ** 1.3)
    cat = N.shape(H, W, lambda dr, k: (dr.ellipse([82 * k, 306 * k, 98 * k, 324 * k], fill=255),
                                       dr.ellipse([84 * k, 297 * k, 94 * k, 307 * k], fill=255),
                                       dr.polygon([(84 * k, 300 * k), (85 * k, 293 * k), (88 * k, 299 * k)], fill=255),
                                       dr.polygon([(90 * k, 299 * k), (93 * k, 293 * k), (94 * k, 301 * k)], fill=255),
                                       dr.arc([92 * k, 308 * k, 104 * k, 326 * k], 270, 90, fill=255, width=2 * k)))
    N.lay(img, cat, INK)
    img[298:300, 86:88] = img[298:300, 90:92] = P.hexrgb("#c8ff5a")  # its eyes
    img[280:324, 71:74] = img[300:303, 40:104] = P.hexrgb("#1a120e")
    shard = N.shape(H, W, lambda dr, k: dr.polygon([(42 * k, 282 * k), (60 * k, 282 * k), (46 * k, 296 * k)], fill=255))
    N.lay(img, shard * 0.85, P.hexrgb("#1a0e0a"))
    for x0, x1 in ((37, 40), (104, 107)):
        img[277:327, x0:x1] = P.hexrgb("#1a120e")
    img[277:280, 37:107] = img[324:327, 37:107] = P.hexrgb("#1a120e")
    img[327:331, 33:111] = P.hexrgb("#3a3440")  # the sill
    for (x0, x1), tilt in (((20, 36), -0.12), ((108, 124), 0.2)):
        sh = N.boards(48, x1 - x0, P.hexrgb("#2e3a36"), P.hexrgb("#0e1412"), bw=5, seed=x0)
        m = N.shape(H, W, lambda dr, k, x0=x0, x1=x1, tilt=tilt: dr.polygon(
            [(x0 * k, 279 * k), (x1 * k, (279 + (x1 - x0) * tilt) * k), (x1 * k, (327 + (x1 - x0) * tilt) * k),
             (x0 * k, 327 * k)], fill=255))
        full = np.zeros_like(img)
        full[279:327 + 4, x0:x1] = np.pad(sh, ((0, 4), (0, 0), (0, 0)), mode="edge")
        N.lay(img, m, full)
    # the door: dark planks under a stone arch, iron bands, light under it
    arch = N.shape(H, W, lambda dr, k: (dr.rectangle([24 * k, 410 * k, 72 * k, 470 * k], fill=255),
                                        dr.ellipse([24 * k, 386 * k, 72 * k, 434 * k], fill=255)))
    door = N.shape(H, W, lambda dr, k: (dr.rectangle([28 * k, 412 * k, 68 * k, 470 * k], fill=255),
                                        dr.ellipse([28 * k, 391 * k, 68 * k, 431 * k], fill=255)))
    N.lay(img, arch, P.hexrgb("#2a2830"))
    planks = np.zeros_like(img)
    planks[386:470, 24:72] = N.boards(84, 48, P.hexrgb("#3e2a1e"), P.hexrgb("#140c08"), bw=9, seed=26)
    N.lay(img, door, planks)
    for y in (418, 452):
        img[y:y + 3, 30:67] = P.hexrgb("#1e1c22")
    img[466:470, 30:66] = P.hexrgb("#ffb050")
    N.lay(img, N.shape(H, W, lambda dr, k: dr.ellipse([56 * k, 432 * k, 63 * k, 439 * k], outline=255, width=k)),
          P.hexrgb("#6a6470"))
    N.glow(img, 86, 404, 40, ORANGE, 0.55)  # the lamp by the door
    img[392:394, 80:92] = INK
    img[394:410, 82:90] = P.hexrgb("#ffc860")
    img[394:410, 82] = img[394:410, 89] = img[401, 82:90] = INK
    # the roof's iron railing, spear-tipped
    rail = N.shape(H, W, lambda dr, k: ([dr.rectangle([0, 241 * k, 196 * k, 243 * k], fill=255),
                                          dr.rectangle([0, 252 * k, 196 * k, 254 * k], fill=255)]
                                         + [(dr.rectangle([x * k, 238 * k, (x + 2) * k, 256 * k], fill=255),
                                             dr.polygon([((x - 1.5) * k, 239 * k), ((x + 1) * k, 233 * k),
                                                         ((x + 3.5) * k, 239 * k)], fill=255))
                                            for x in range(4, 196, 10)]))
    N.lay(img, rail, P.hexrgb("#141018"))


def _tower(img) -> None:
    """The round stone tower on the right of the buttons: battlements, moss and ivy, glowing slits."""
    x0, x1 = 296, LEFT
    body = N.stones(450, x1 - x0, P.hexrgb("#6a6676"), P.hexrgb("#1c1a22"), bw=24, bh=15, seed=31, moss=0.6)
    u = np.linspace(0, 1, x1 - x0)
    light = np.clip(np.cos((u - 0.34) * np.pi * 0.95), 0.12, 1) ** 0.8 * 0.9 + 0.1  # round: lit from the moon's side
    img[44:470, x0:x1] = body[24:] * light[None, :, None]
    for k, x in enumerate(range(x0, x1, 22)):  # battlements, on a corbelled ring
        img[20:44, x:min(x1, x + 13)] = body[:24, x - x0:min(x1, x + 13) - x0] * light[None, x - x0:min(x1, x + 13) - x0,
                                                                                  None]
        img[20:22, x:min(x1, x + 13)] *= 1.3
    img[44:50, x0:x1] = img[44:50, x0:x1] * 0.45
    img[50:52, x0:x1] = INK
    img[20:470, x0:x0 + 2] = INK
    for x, y in ((372, 224), (372, 336)):  # arrow slits, lit from inside
        N.glow(img, x + 4, y + 12, 34, ORANGE, 0.45)
        slit = N.shape(H, W, lambda d, k, x=x, y=y: (d.rectangle([x * k, (y + 4) * k, (x + 8) * k, (y + 26) * k],
                                                                  fill=255),
                                                      d.polygon([(x * k, (y + 5) * k), ((x + 4) * k, (y - 2) * k),
                                                                 ((x + 8) * k, (y + 5) * k)], fill=255)))
        N.lay(img, slit, P.hexrgb("#ffb040"))
    rng = np.random.default_rng(32)

    def ivy(d, k):  # creeping up from the foot, on the dark side
        for v in range(7):
            x, y = 298 + rng.uniform(0, 26), 470.0
            for _ in range(int(rng.uniform(16, 34))):
                nx, ny = x + rng.uniform(-3.5, 4.0), y - rng.uniform(3, 7)
                d.line([(x * k, y * k), (nx * k, ny * k)], fill=255, width=k)
                for _ in range(2):
                    lx, ly = nx + rng.uniform(-5, 5), ny + rng.uniform(-3, 3)
                    s = rng.uniform(1.6, 3.2)
                    d.ellipse([(lx - s) * k, (ly - s * 0.8) * k, (lx + s) * k, (ly + s * 0.8) * k], fill=255)
                x, y = nx, ny

    m = N.shape(H, W, ivy)
    leaves = P.mix(P.hexrgb("#0e1a0a"), P.hexrgb("#2e4a1c"), N.field(H, W, 4, 33))
    N.lay(img, m, leaves)


def _ground(img) -> None:
    """Dark earth and dead grass, a cobbled path, gravestones and jack-o'-lanterns, mist low over it."""
    g = N.field(130, LEFT, 10, 41)
    earth = P.mix(P.hexrgb("#0c0a08"), P.hexrgb("#2a2416"), g)
    img[470:600, :LEFT] = earth
    rng = np.random.default_rng(42)

    def path(d, k):
        top, bottom = (248, 22), (236, 74)
        pts_l, pts_r = [], []
        for t in np.linspace(0, 1, 20):
            cx = top[0] + (bottom[0] - top[0]) * t + math.sin(t * 3.0) * 12
            half = top[1] + (bottom[1] - top[1]) * t
            y = 470 + 130 * t
            pts_l.append(((cx - half) * k, y * k))
            pts_r.append(((cx + half) * k, y * k))
        d.polygon(pts_l + pts_r[::-1], fill=255)

    cobbles = N.stones(130, LEFT, P.hexrgb("#4a4650"), P.hexrgb("#141218"), bw=14, bh=8, seed=43, moss=0.4)
    full = np.zeros_like(img)
    full[470:600, :LEFT] = cobbles
    N.lay(img, N.shape(H, W, path) * 0.9, full)

    def grass(d, k):
        for _ in range(900):
            x, y = rng.uniform(0, LEFT), rng.uniform(472, 600)
            h = rng.uniform(3, 9)
            d.line([(x * k, y * k), ((x + rng.uniform(-3, 3)) * k, (y - h) * k)], fill=255, width=k)

    N.lay(img, N.shape(H, W, grass) * 0.85, P.hexrgb("#4a4224"))
    img[470:473, :LEFT] = img[470:473, :LEFT] * 0.5
    for kind, cx, foot, h in ((0, 152, 514, 40), (2, 46, 506, 36), (1, 330, 566, 58), (0, 196, 598, 34),
                              (2, 356, 500, 30)):
        thing = N.model(T.gravestone(kind, seed=cx), height=h, elevation=8)
        N.over(img, thing, cx - thing.shape[1] // 2, foot - thing.shape[0])
    for seed, cx, foot, h in ((1, 104, 526, 24), (2, 292, 592, 28), (3, 272, 508, 18), (4, 178, 560, 20)):
        thing = N.model(T.jack(seed), height=h)
        N.glow(img, cx, foot - h // 2, h * 2.2, ORANGE, 0.5)
        N.over(img, thing, cx - thing.shape[1] // 2, foot - thing.shape[0])
    N.fog(img[:, :LEFT], 500, 600, colour=P.hexrgb("#4a5068"), strength=0.3, seed=6)


def _wall(img) -> None:
    """The right: a dark stone wall, cobwebs in its corners, and the game's description in a dark tooltip."""
    wall = N.stones(H, W - LEFT, P.hexrgb("#2a2232"), P.hexrgb("#0a080c"), bw=46, bh=26, seed=51, moss=0.2)
    yy, xx = np.mgrid[0:H, 0:W - LEFT]
    vignette = 1 - 0.45 * np.clip(np.hypot((xx - 185) / 260, (yy - 300) / 360), 0, 1)
    img[:, LEFT:] = wall * vignette[..., None]
    img[:58, LEFT:] *= 0.55
    img[58:60, LEFT:] = INK
    img[60:61, LEFT:] = P.hexrgb("#7a3a10")
    img[:, LEFT:LEFT + 2] = INK
    webs = N.shape(H, W, lambda d, k: (N.web(d, 799, 61, 92, k, math.pi / 2, math.pi, 7, 6),
                                       N.web(d, 432, 61, 58, k, 0, math.pi / 2, 5, 4),
                                       d.line([(780 * k, 61 * k), (780 * k, 150 * k)], fill=255, width=k)))
    N.lay(img, webs * 0.45, P.hexrgb("#b8b4c4"))
    spider = N.shape(H, W, lambda d, k: ([d.ellipse([776 * k, 148 * k, 784 * k, 158 * k], fill=255),
                                           d.ellipse([777.5 * k, 144 * k, 782.5 * k, 150 * k], fill=255)]
                                          + [d.line([(780 * k, 153 * k), ((780 + sx * 7) * k, (153 + dy) * k),
                                                     ((780 + sx * 9) * k, (153 + dy + 5) * k)], fill=255, width=k)
                                             for sx in (-1, 1) for dy in (-5, -2, 1, 4)]))
    N.lay(img, spider, INK)
    x0, y0, x1, y1 = DESCRIPTION
    img[y0:y1, x0:x1] = P.hexrgb(TOOLTIP[0])
    for t, row in enumerate(range(y0 + 1, y1 - 1)):
        img[row, x0 + 1] = img[row, x1 - 2] = P.mix(P.hexrgb(TOOLTIP[1]), P.hexrgb(TOOLTIP[2]), t / (y1 - y0))
    img[y0 + 1, x0 + 1:x1 - 1] = P.hexrgb(TOOLTIP[1])
    img[y1 - 2, x0 + 1:x1 - 1] = P.hexrgb(TOOLTIP[2])
    img[y0, x0:x1] = img[y1 - 1, x0:x1] = INK
    img[y0:y1, x0] = img[y0:y1, x1 - 1] = INK


def _sign(img, masks, name, box) -> None:
    """A board of dark old planks where the game writes a button's name (in white): light along its top, shade
    along its bottom, a black edge."""
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = N.boards(y1 - y0, x1 - x0, P.hexrgb(BUTTON["fill"]), P.hexrgb(BUTTON["shade"]), bw=7,
                                 seed=zlib.crc32(name.encode()) % 1000, across=True) * 0.8
    img[y0 + 1, x0 + 1:x1 - 1] = P.hexrgb(BUTTON["light"])
    img[y0 + 1:y1 - 1, x0 + 1] = P.hexrgb(BUTTON["light"]) * 0.8
    img[y1 - 2, x0 + 1:x1 - 1] = P.hexrgb(BUTTON["shade"]) * 0.6
    edge = P.hexrgb(BUTTON["edge"])
    img[y0, x0:x1] = img[y1 - 1, x0:x1] = edge
    img[y0:y1, x0] = img[y0:y1, x1 - 1] = edge
    if masks is not None:
        _mark(masks, name, box)


def _mark(masks, name, box) -> None:
    x0, y0, x1, y1 = box
    masks.setdefault(name, np.zeros((H, W), bool))[y0:y1, x0:x1] = True


def _place(img, masks, name, rgba, cx: float, top: int = None, bottom: int = None) -> None:
    """A rendered object, centred on cx, its top or its foot at the given height; it joins the button's mask."""
    h, w = rgba.shape[:2]
    y = top if top is not None else bottom - h
    N.over(img, rgba, int(round(cx - w / 2)), y, masks, name)


def _banner(img, masks) -> None:
    """Learn to Play: a tattered purple banner on an iron bar, a jack-o'-lantern and a spell book on it."""
    img[36:238, 110:116] = N.boards(202, 6, P.hexrgb("#3a2a22"), P.hexrgb("#120c0a"), bw=6, seed=61)
    img[36:238, 110] = img[36:238, 115] = INK
    bar = N.shape(H, W, lambda d, k: (d.rectangle([8 * k, 32 * k, 114 * k, 37 * k], fill=255),
                                      d.ellipse([4 * k, 30 * k, 12 * k, 39 * k], fill=255)))
    N.lay(img, bar, P.hexrgb("#24222a"))
    masks.setdefault("learn", np.zeros((H, W), bool))[bar > 0.5] = True
    rng = np.random.default_rng(62)
    edge = [(14, 37), (108, 37), (108, 168)]
    x = 108.0
    while x > 14:
        x2 = max(14.0, x - rng.uniform(6, 13))
        edge += [((x + x2) / 2, rng.uniform(176, 187)), (x2, rng.uniform(160, 170))]
        x = x2
    cloth = N.shape(H, W, lambda d, k: d.polygon([(px * k, py * k) for px, py in edge], fill=255))
    xx = np.arange(W)[None, :]
    folds = 0.72 + 0.28 * np.sin((xx - 14) / 94 * np.pi * 3.0) ** 2
    weave = N.field(H, W, 2.0, 63, 2) * 0.25 + N.field(H, W, 16, 64) * 0.35
    colour = P.mix(P.hexrgb("#14061c"), P.hexrgb("#5a2272"), np.clip(folds * (0.55 + weave), 0, 1))
    N.lay(img, cloth, colour)
    img[40:44, 14:108] = P.hexrgb("#c86a14") * 0.85  # an orange stripe along the top
    img[44:45, 14:108] *= 0.5
    masks["learn"] |= cloth > 0.5
    face = N.model(T.jack(7, radius=4.0), width=56)
    _place(img, masks, "learn", face, 61, top=52)
    book = N.model(T.book(), width=64, elevation=40)
    _place(img, masks, "learn", book, 61, top=122)


def _map(img, masks) -> None:
    """Map Editor: an old map in a dark frame on a post: a dotted way past a skull to a red cross."""
    img[338:470, 224:231] = N.boards(132, 7, P.hexrgb("#3a2a22"), P.hexrgb("#120c0a"), bw=7, seed=71)
    img[338:470, 224] = img[338:470, 230] = INK
    x0, y0, x1, y1 = 205, 298, 251, 340
    img[y0:y1, x0:x1] = N.boards(y1 - y0, x1 - x0, P.hexrgb("#3a2618"), P.hexrgb("#120a06"), bw=6, seed=72)
    px0, py0, px1, py1 = x0 + 5, y0 + 5, x1 - 5, y1 - 5
    yy, xx = np.mgrid[py0:py1, px0:px1]
    burn = np.clip(1 - np.minimum.reduce([xx - px0, px1 - 1 - xx, yy - py0, py1 - 1 - yy]) / 5, 0, 1)
    paper = P.mix(P.hexrgb("#e2cc94"), P.hexrgb("#8a6a3a"), N.field(py1 - py0, px1 - px0, 6, 73) * 0.6 + burn * 0.6)
    img[py0:py1, px0:px1] = paper
    ink = N.shape(H, W, lambda d, k: ([d.ellipse([(px0 + 3 + 3 * j) * k, (py1 - 6 - 2.4 * j) * k,
                                                  (px0 + 4 + 3 * j) * k, (py1 - 5 - 2.4 * j) * k], fill=255)
                                       for j in range(9)]
                                      + [d.polygon([(px0 + 6 * k // k) * k, (py0 + 10) * k] and
                                                   [((px0 + 4) * k, (py0 + 12) * k), ((px0 + 8) * k, (py0 + 5) * k),
                                                    ((px0 + 12) * k, (py0 + 12) * k)], fill=255),
                                         d.polygon([((px0 + 10) * k, (py0 + 13) * k), ((px0 + 15) * k, (py0 + 6) * k),
                                                    ((px0 + 20) * k, (py0 + 13) * k)], fill=255)]))
    N.lay(img, ink * 0.8, P.hexrgb("#3a2414"))
    cross = N.shape(H, W, lambda d, k: (d.line([((px1 - 12) * k, (py0 + 5) * k), ((px1 - 5) * k, (py0 + 12) * k)],
                                               fill=255, width=2 * k),
                                        d.line([((px1 - 5) * k, (py0 + 5) * k), ((px1 - 12) * k, (py0 + 12) * k)],
                                               fill=255, width=2 * k)))
    N.lay(img, cross, P.hexrgb("#b0181a"))
    skull = N.shape(H, W, lambda d, k: d.ellipse([(px0 + 22) * k, (py1 - 15) * k, (px0 + 28) * k, (py1 - 9) * k],
                                                 fill=255))
    N.lay(img, skull, P.hexrgb("#f2ead2"))
    img[py1 - 13, px0 + 23:px0 + 25] = img[py1 - 13, px0 + 26:px0 + 28] = INK
    for x, y in ((x0 + 2, y0 + 2), (x1 - 4, y0 + 2), (x0 + 2, y1 - 4), (x1 - 4, y1 - 4)):
        img[y:y + 2, x:x + 2] = P.hexrgb("#8a8a92")
    img[y0, x0:x1] = img[y1 - 1, x0:x1] = INK
    img[y0:y1, x0] = img[y0:y1, x1 - 1] = INK
    _mark(masks, "map", (x0, y0, x1, y1))


def _exit(img, masks) -> None:
    """Exit: an arrow-shaped board of dark planks pointing left, on a post (the game writes "Exit" on it)."""
    img[594:600, 118:126] = P.hexrgb("#2a1e16")
    ys, xs = np.mgrid[0:H, 0:W]
    arrow = (xs >= 46) & (xs < 158) & (ys >= 552) & (ys < 594)
    arrow |= (xs >= 10) & (xs < 46) & (np.abs(ys - 573) <= (xs - 10) * 0.9)
    planks = np.zeros_like(img)
    planks[540:600, 0:170] = N.boards(60, 170, P.hexrgb(BUTTON["fill"]), P.hexrgb(BUTTON["shade"]), bw=8, seed=81,
                                      across=True) * 0.85
    img[arrow] = planks[arrow]
    rim = arrow & _outline(~arrow, 1)
    bevel = arrow & ~rim & _outline(rim | ~arrow, 1)
    img[bevel & (ys < 573)] = P.hexrgb(BUTTON["light"])
    img[bevel & (ys >= 573)] = P.hexrgb(BUTTON["shade"]) * 0.6
    img[rim] = P.hexrgb(BUTTON["edge"])
    for x in (54, 150):
        img[557:559, x:x + 2] = img[587:589, x:x + 2] = P.hexrgb("#8a8a92")
    masks["exit"] = arrow


def scene() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The 800x600 menu picture (RGB 0..1) and each button's mask (its object and the board its name is on).

    A Halloween night on the left, where the buttons are: a full moon behind the title, a haunted house with a lit
    window and a railed roof, a round stone tower, a churchyard; every button name on a dark wooden board. On the
    right, where the game opens the Single Player menu, a dark stone wall with cobwebs, and the game's description
    of a button in a dark tooltip with an orange edge."""
    img = np.zeros((H, W, 3))
    masks: dict[str, np.ndarray] = {}
    _scene_far(img)
    _house(img)
    _tower(img)
    _ground(img)
    _wall(img)
    # Single Player: the jack-o'-lantern hanging on a chain under its board, lighting the tower
    lantern = N.model(T.lantern(), width=100, elevation=12)
    N.glow(img, 367, 62 + lantern.shape[0] * 0.55, 92, ORANGE, 0.5)
    chain = N.shape(H, W, lambda d, k: [d.ellipse([(364 if j % 2 else 365.5) * k, (36 + 6 * j) * k,
                                                   (370 if j % 2 else 368.5) * k, (44 + 6 * j) * k],
                                                  outline=255, width=max(1, int(1.6 * k)))
                                        for j in range(5)])
    N.lay(img, chain, P.hexrgb("#4a4652"))
    masks.setdefault("single", np.zeros((H, W), bool))[chain > 0.5] = True
    _place(img, masks, "single", lantern, 367, top=62)
    # History: the spell book on its lectern, on the roof
    N.glow(img, 112, 206, 30, ORANGE, 0.35)
    N.glow(img, 196, 198, 30, ORANGE, 0.35)
    _place(img, masks, "history", N.model(T.lectern(), width=92, elevation=25), 154, bottom=258)
    # Multiplayer: the skull shield on its scythes, on the tower
    _place(img, masks, "multi", N.model(T.shield(), width=92), 311, top=244)
    # Options: the cauldron, on a stone block in front of the house
    img[433:470, 118:174] = N.stones(37, 56, P.hexrgb("#5a5664"), P.hexrgb("#1a1820"), bw=28, bh=12, seed=91)
    img[433, 118:174] = img[469, 118:174] = INK
    img[433:470, 118] = img[433:470, 173] = INK
    N.glow(img, 146, 392, 56, GREEN, 0.45)
    _place(img, masks, "options", N.model(T.cauldron(), width=70), 146, bottom=434)
    # Zone: the crystal ball, on a ledge of the tower
    img[426:432, 282:330] = N.stones(6, 48, P.hexrgb("#6a6676"), P.hexrgb("#1c1a22"), bw=16, bh=6, seed=92)
    img[431, 282:330] = INK
    N.glow(img, 305, 408, 40, P.hexrgb("#b07aff"), 0.55)
    _place(img, masks, "zone", N.model(T.crystal_ball(), height=40), 305, bottom=427)
    _map(img, masks)
    _banner(img, masks)
    _exit(img, masks)
    for name, box in SIGNS.items():
        _sign(img, masks, name, box)
    # the title, over the moon
    title = N.title(210, 92)
    tx, ty = 108, 36 + (111 - title.shape[0]) // 2
    N.over(img, title, tx, ty)
    masks["logo"] = np.zeros((H, W), bool)
    masks["logo"][ty:ty + title.shape[0], tx:tx + title.shape[1]] = title[..., 3] > 0.5
    masks["banner"] = masks["logo"] | masks["learn"] | masks["single"]
    return np.clip(img, 0, 1), masks


# --------------------------------------------------------------------------- the pictures the game uses


def _outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    grown = mask.copy()
    for _ in range(width):
        g = grown.copy()
        g[1:] |= grown[:-1]
        g[:-1] |= grown[1:]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    return grown & ~mask


def glow_colour(original: slp.SlpFrame, background: slp.SlpFrame, at: tuple[int, int], palette: np.ndarray,
                inside: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """The colour of the glow a button picture draws around its object (None if it has none). `inside` is the
    object's own shape: a picture that only lights the object up has no glow."""
    x, y = at
    px = original.pixels
    h, w = px.shape
    bg = background.pixels[y:y + h, x:x + w]
    both = (px >= 0) & (px < 256) & (bg >= 0) & (bg < 256)
    a, b = palette[np.where(both, px, 0)].astype(float), palette[np.where(both, bg, 0)].astype(float)
    bright = both & (np.abs(a - b).sum(-1) > 100) & (a.sum(-1) > 450)
    if inside is not None:
        bright &= ~(inside | _outline(inside, 1))
    if bright.sum() < 20:
        return None
    return np.median(a[bright], axis=0) / 255


def _cut(f: slp.SlpFrame) -> bool:
    """A cut-out picture: the object alone, much of it transparent (the others only have a few clear corners)."""
    return float((f.pixels < 0).mean()) > 0.1


def pictures(original: list[slp.SlpFrame], palette: np.ndarray, quant: Quantiser) -> list[slp.SlpFrame]:
    """Every picture of the menu SLP redrawn: the new background, and each button's pictures cut from it at the
    same place, with the glow of the original around the new object. Pictures it doesn't know are kept."""
    img, masks = scene()
    codes = quant.indices(np.clip(img * 255 + 0.5, 0, 255).astype(np.int64).reshape(-1, 3)).reshape(H, W)
    out = list(original)
    out[0] = slp.SlpFrame(codes.astype(np.int16), original[0].hotspot)
    for name, (frames, cut_at, rect_at) in BUTTONS.items():
        mask = masks[name]
        silhouette = None  # the original object's shape, from its cut-out picture, where the others are
        if cut_at and rect_at and frames[0] < len(original) and _cut(original[frames[0]]):
            c = original[frames[0]].pixels >= 0
            silhouette = np.zeros(original[frames[1]].pixels.shape, bool) if len(frames) > 1 else None
            if silhouette is not None:
                dx, dy = cut_at[0] - rect_at[0], cut_at[1] - rect_at[1]
                hh, ww = min(c.shape[0], silhouette.shape[0] - dy), min(c.shape[1], silhouette.shape[1] - dx)
                silhouette[dy:dy + hh, dx:dx + ww] = c[:hh, :ww]
        for k in frames:
            if k >= len(original):
                continue
            f = original[k]
            h, w = f.pixels.shape
            cut = _cut(f)
            alone = rect_at is None  # a picture of the object only (the logo): no glow
            x, y = cut_at if (cut and cut_at) or alone else rect_at
            picture = img[y:y + h, x:x + w].copy()
            shape = mask[y:y + h, x:x + w]
            inside = silhouette if silhouette is not None and silhouette.shape == (h, w) else None
            plain = cut or alone or name in DARKER
            colour = None if plain else glow_colour(f, original[0], (x, y), palette, inside)
            if colour is not None:
                picture[_outline(shape)] = colour
            if name in DARKER and not cut and k in frames[1:3]:  # the thing darker, its name's button as it is
                thing = shape.copy()
                if name in SIGNS:
                    bx0, by0, bx1, by1 = SIGNS[name]
                    thing[max(0, by0 - y):max(0, by1 - y), max(0, bx0 - x):max(0, bx1 - x)] = False
                picture[thing] *= DARKER[name][frames.index(k) - 1]
            px = quant.indices(np.clip(picture * 255 + 0.5, 0, 255).astype(np.int64).reshape(-1, 3)).reshape(h, w)
            px = px.astype(np.int16)
            px[(~shape) if cut else (f.pixels < 0)] = slp.TRANSPARENT
            out[k] = slp.SlpFrame(px, f.hotspot)
    return out


def known(sizes: list[tuple[int, int]]) -> bool:
    """True if a menu SLP has the pictures this module knows, at their sizes."""
    return len(sizes) > 52 and all(sizes[k] == wh for k, wh in SIZES.items())
