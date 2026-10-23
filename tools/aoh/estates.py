"""Every economic and military building, made from the haunted culture kits (haunted.py), as Parts centred on their
footprints (2x2 = 80 units, 3x3 = 128, 4x4 = 176). Each says what it is (a barracks has a training yard with
scarecrow dummies, a market its stalls, a stable its paddock); its culture says how it is built.

The animated buildings take (t, part) as Age of Minecraft's did (props.py "overlay" mode): part "body" is the
building alone, "all" adds what moves at time t (the mill's sails, the forge's smoke, the university's orrery).
"""
from __future__ import annotations

import math

import numpy as np

from . import haunted as H
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .geometry import Part
from .haunted import FOOT, box, hull


def _k(style: str, age: int) -> H.Kit:
    return H.kit("G" if style == "G" else style, age if style != "G" else 1)


def _part(name, shapes, *children) -> Part:
    return Part(name, boxes=list(shapes)).add(*children)


def _cottage(k: H.Kit, x0, y0, x1, y1, h, door_face="+y", axis="x", door_at=None, z0=2.0) -> tuple[list, float]:
    """A small hall with its roof: the walls (windows on the camera sides, a door) and the kit's roof."""
    if door_at is None:
        door_at = (x0 + x1) / 2 if door_face == "+y" else (y0 + y1) / 2
    out = H.walls(k, x0, y0, x1, y1, z0, h, door=(door_face, door_at, 7.0, min(14.0, h * 0.62)), every=14.0)
    roof, top = H.roof_for(k, x0, x1, y0, y1, z0 + h, axis)
    return out + roof, top


# --------------------------------------------------------------------------- house

def house(style: str, age: int, variant: int = 0) -> Part:
    """A haunted home on a 2x2 footprint, in three variants (the game's three house pictures)."""
    k = _k(style, age)
    v = variant % 3
    out = []
    if k.key == "G":  # a crooked log cabin, thatched, a lantern by the door
        cab, top = _cottage(k, -26, -22, 22, 20, 24, "+y" if v != 1 else "+x", "x" if v != 2 else "y")
        out += cab
        out += H.door_light(10.0, 21.5, 14.0, k)
        if v == 2:
            out += H.pumpkins(14.0, 26.0, 0.0, 2, seed=v)
        out += H.wall_banner(23.0, -8.0, 22.0, "+x", 4.0, 10.0)
        return _part("house", out)
    if k.key == "W":
        if v == 0:  # a crooked cottage under a steep gable, a witch-hat turret on its corner
            cab, top = _cottage(k, -28, -22, 18, 18, 28, "+y", "x", door_at=-6.0)
            out += cab + H.tower(k, 20.0, 20.0, 7.0, 0.0, 40.0, seed=v)
        elif v == 1:  # two storeys, the gable facing the street, a lit attic window
            cab, top = _cottage(k, -24, -26, 20, 20, 36, "+x", "y")
            out += cab
        else:  # a cottage and a leaning chimney
            cab, top = _cottage(k, -26, -22, 22, 20, 26, "+y", "x")
            chim = P.bricks(P.hexrgb("#5e3e34"), P.hexrgb("#2a1a14"), 2.4, 1.2, seed=7)
            out += cab + [box((-20, -14, 20), (-13, -7, top + 8), chim)]
        out += H.pumpkins(4.0, 26.0, 0.0, 2, seed=v) + H.door_light(-14.0, 21.0, 16.0, k)
        out += H.bats([(-6.0, -4.0, top + 10), (6.0, 2.0, top + 16)], seed=v) if v == 1 else []
        out += H.wall_banner(19.0 if v != 1 else 21.0, -10.0, 24.0, "+x", 4.0, 10.0)
    elif k.key == "E":
        if v == 0:  # a narrow stone house under a needle-steep roof, a spire at the ridge
            cab, top = _cottage(k, -24, -20, 18, 18, 30, "+y", "x")
            out += cab + H.spire(-3.0, -1.0, top - 6.0, 2.4, 18.0, k.roof)
        elif v == 1:
            cab, top = _cottage(k, -20, -26, 18, 20, 34, "+x", "y")
            out += cab + H.tower(k, -20.0, 20.0, 6.0, 0.0, 34.0, seed=v)
        else:
            cab, top = _cottage(k, -26, -22, 22, 20, 26, "+y", "x")
            out += cab + H.bats([(-8.0, 0.0, top + 8), (10.0, -6.0, top + 14), (0.0, 8.0, top + 20)], seed=v)
        out += H.door_light(-14.0, 21.0, 16.0, k) + H.wall_banner(19.0, -10.0, 26.0, "+x", 4.0, 12.0)
        out += [*H.gravestone(22.0, 30.0, 0.0, 5.0, v), *H.gravestone(28.0, 22.0, 0.0, 4.0, v + 1)]
    elif k.key == "M":
        if v == 0:  # a flat-roofed tomb house with a dome over its upper room
            out += H.walls(k, -26, -24, 20, 20, 2.0, 30, door=("+y", -4.0, 7.0, 15.0))
            out += H.flat_roof(k, -26, -24, 20, 20, 32.0)
            out += H.dome(-2.0, -2.0, 33.0, 10.0 if k.age >= 3 else 8.0, k.roof, onion=k.age >= 4)
        elif v == 1:  # two blocks, the upper one smaller, a hieroglyph band
            out += H.walls(k, -26, -24, 20, 20, 2.0, 22, door=("+y", 0.0, 7.0, 14.0))
            out += [box((-26.5, -24.5, 22.0), (20.5, 20.5, 26.0), k.upper)]
            out += H.walls(k, -24, -22, 4, 4, 26.0, 16, plinth=0.0) + H.flat_roof(k, -24, -22, 4, 4, 42.0)
        else:  # a wind tower beside the house
            out += H.walls(k, -26, -24, 20, 20, 2.0, 26, door=("+y", 0.0, 7.0, 14.0)) + H.flat_roof(k, -26, -24, 20,
                                                                                                     20, 28.0)
            out += H.walls(k, -26, -24, -14, -12, 28.0, 22, every=6.0, plinth=0.0) + H.flat_roof(k, -26, -24, -14,
                                                                                                -12, 50.0)
        out += H.urn(24.0, 26.0, 0.0, 5.0) + H.urn(14.0, 28.0, 0.0, 4.0) + H.door_light(-12.0, 21.0, 15.0, k)
        out += H.wall_banner(21.0, -10.0, 26.0, "+x", 4.0, 10.0)
    elif k.key == "F":
        out += [box((-30, -26, 0), (24, 22, 4.0), k.base)]  # a raised stone platform
        if v == 0:
            out += H.walls(k, -24, -20, 18, 16, 4.0, 22, door=("+y", -4.0, 8.0, 15.0))
            out += H.pagoda_roof(k, -24, 18, -20, 16, 26.0, 16.0, over=7.0)
        elif v == 1:  # two tiers
            out += H.walls(k, -24, -20, 18, 16, 4.0, 18, door=("+x", -2.0, 8.0, 14.0))
            out += H.pagoda_roof(k, -24, 18, -20, 16, 22.0, 8.0, over=6.0)
            out += H.walls(k, -16, -12, 10, 8, 26.0, 12, plinth=0.0) + H.pagoda_roof(k, -16, 10, -12, 8, 38.0, 12.0)
        else:
            out += H.walls(k, -24, -20, 18, 16, 4.0, 20, door=("+y", 2.0, 8.0, 14.0))
            out += H.pagoda_roof(k, -24, 18, -20, 16, 24.0, 14.0, over=7.0)
            out += [S.rod((22.0, 22.0, 0.0), (22.0, 22.0, 14.0), 0.5, k.trim)]
        out += H.lanterns_hanging([(20.0, 18.0, 20.0), (-10.0, 24.0, 20.0)], k)
        out += H.wall_banner(19.0, -8.0, 22.0, "+x", 4.0, 10.0)
    else:  # X
        if v == 0 or k.age == 2:  # a painted hut, thatch or a flat terrace roof
            out += H.walls(k, -24, -20, 18, 18, 2.0, 24, door=("+y", -2.0, 7.0, 14.0))
            roof, top = H.roof_for(k, -24, 18, -20, 18, 26.0)
            out += roof
            if v == 1:
                out += [S.rod((0.0, 0.0, 26.0), (0.0, 0.0, 34.0), 4.0, k.roof, 7.0)]
        elif v == 1:  # a round painted hut
            out += [S.rod((-4.0, -2.0, 2.0), (-4.0, -2.0, 26.0), 18.0, k.wall), S.rod((-4.0, -2.0, 0.0), (-4.0, -2.0,
                                                                                                          2.5), 19.0,
                                                                                    k.base)]
            out += [S.rod((-4.0, -2.0, 26.0), (-4.0, -2.0, 46.0), 21.0, H.thatch(P.hexrgb("#7a6a40"), seed=3), 0.5)]
            out += H.walls(k, -6, 14, 2, 16.5, 2.0, 0.1, door=None, windows=False, plinth=0.0)
        else:
            out += H.walls(k, -24, -20, 18, 18, 2.0, 20, door=("+y", -2.0, 7.0, 14.0))
            out += H.terrace(k, -24, -20, 18, 18, 22.0, steps=2)
            out += H.walls(k, -16, -12, 4, 6, 30.0, 12, plinth=0.0) + H.terrace(k, -16, -12, 4, 6, 42.0, steps=1)
        out += H.marigolds(8.0, 24.0, 0.0, 6, seed=v) + H.candles(-12.0, 24.0, 0.0, 3, seed=v)
        out += PK.lantern((16.0, 26.0, 0.0), 2.2, seed=v)
        out += H.wall_banner(19.0, -8.0, 22.0, "+x", 4.0, 10.0)
    return _part("house", out)


# --------------------------------------------------------------------------- the big halls

def _civic(k: H.Kit, x0, y0, x1, y1, h, door_face="+y", towers=(), axis="x", seed=0) -> tuple[list, float]:
    """A large hall in the culture's manner, with towers at the given (x, y, r, h) spots."""
    out = []
    door_c = (x0 + x1) / 2 if door_face == "+y" else (y0 + y1) / 2
    if k.key == "M":
        out += H.walls(k, x0, y0, x1, y1, 3.0, h, door=(door_face, door_c, 10.0, h * 0.55))
        out += [box((x0 - 0.5, y0 - 0.5, 3.0 + h * 0.7), (x1 + 0.5, y1 + 0.5, 3.0 + h * 0.8), k.upper)]
        out += H.flat_roof(k, x0, y0, x1, y1, 3.0 + h)
        out += H.dome((x0 + x1) / 2, (y0 + y1) / 2, 4.0 + h, min(x1 - x0, y1 - y0) * 0.3, k.roof, onion=k.age >= 4)
        top = 4.0 + h + min(x1 - x0, y1 - y0) * 0.3
    elif k.key == "X" and k.age >= 3:
        steps = 3
        for i in range(steps):
            a = 6.0 * i
            out.append(box((x0 + a, y0 + a, i * h / steps), (x1 - a, y1 - a, (i + 1) * h / steps),
                           k.wall if i % 2 == 0 else k.base))
        out.append(box((x0 + 18 - 0.5, (y0 + y1) / 2 - 2, 0), (x1 - 18 + 0.5, y1 + 2, 4.0), k.base))  # the stair
        sx0, sy0, sx1, sy1 = x0 + 20, y0 + 20, x1 - 20, y1 - 20
        temple = H.walls(k, sx0, sy0, sx1, sy1, h, h * 0.5, door=("+y", (sx0 + sx1) / 2, 7.0, h * 0.35), every=12.0,
                         plinth=0.0)
        out += temple + H.terrace(k, sx0, sy0, sx1, sy1, h * 1.5, steps=2, step=3.0, inset=3.0)
        top = h * 1.5 + 6.0
    else:
        z0 = 3.0
        out += [box((x0 - 2, y0 - 2, 0), (x1 + 2, y1 + 2, z0), k.base)]
        out += H.walls(k, x0, y0, x1, y1, z0, h, door=(door_face, door_c, 10.0, min(h * 0.55, 20.0)), plinth=0.0)
        if k.key == "F":
            out += H.pagoda_roof(k, x0, x1, y0, y1, z0 + h, min(x1 - x0, y1 - y0) * 0.35, over=9.0)
            top = z0 + h + min(x1 - x0, y1 - y0) * 0.35
        else:
            roof, top = H.roof_for(k, x0, x1, y0, y1, z0 + h, axis, scale=0.8)
            out += roof
    for (tx, ty, tr, th) in towers:
        out += H.tower(k, tx, ty, tr, 0.0, th, seed=seed)
    return out, top


def town_center(style: str, age: int) -> Part:
    """The Town Center (4x4): the village's great hall and its tower, flags and a graveyard by the gate."""
    k = _k(style, age)
    out, top = _civic(k, -58, -58, 34, 30, 44 if k.key != "G" else 36,
                      towers=[(-54.0, 40.0, 11.0, 70.0)] if k.key not in ("X", "M") or k.age < 3 else
                      [(42.0, -50.0, 10.0, 66.0)], seed=1)
    if k.key == "G":
        out += [box((-62, -62, 0), (38, 34, 2.0), k.base)]
    out += H.wall_banner(35.0, -30.0, 40.0, "+x", 6.0, 16.0) + H.wall_banner(35.0, 4.0, 40.0, "+x", 6.0, 16.0)
    out += H.wall_banner(-30.0, 31.0, 40.0, "+y", 6.0, 16.0)
    out += H.team_flag((40.0, 36.0, 0.0), 30.0, 2) + H.team_flag((-60.0, -60.0, top - 10), 20.0, 3)
    for i, (x, y) in enumerate(((52.0, 42.0), (60.0, 30.0), (46.0, 56.0))):
        out += H.gravestone(x, y, 0.0, 6.0, i, seed=i)
    if k.key == "M":
        out += H.sarcophagus(38.0, 10.0, 0.0, "x") + H.urn(40.0, -10.0, 0.0, 6.0)
    elif k.key == "X":
        out += H.marigolds(44.0, 36.0, 0.0, 9) + H.candles(36.0, 44.0, 0.0, 5)
    elif k.key == "F":
        out += H.lanterns_hanging([(36.0, -50.0, 30.0), (36.0, 22.0, 30.0), (-50.0, 32.0, 30.0)], k)
    else:
        out += H.cauldron(48.0, 4.0, 0.0, 4.0) + H.pumpkins(20.0, 40.0, 0.0, 3)
    return _part("town_center", out)


# --------------------------------------------------------------------------- mill, smith, university (animated)

def _sails(k: H.Kit, hub, t: float, r: float = 34.0) -> Part:
    """Four tattered sails turning on the hub (facing +x+y), in rotting canvas; frame spins with t."""
    hub = np.asarray(hub, float)
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1, seed=3)

    def torn(look):
        p = look.p
        holes = P.noise(p, 2.4, 5) > 0.66
        return np.where(holes, 0.0, 1.0)
    canvas = S.Material(color=lambda l: P.mix(P.hexrgb("#8a8270"), P.hexrgb("#3a362e"),
                                              0.4 * P.fbm(l.p, 2.0, 6) + 0.5 * (1 - torn(l))))
    shapes = [S.ball(tuple(hub), 2.2, P.metal(P.hexrgb("#2c2a32")))]
    d = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)  # the axle points out to the camera's side
    u = np.array([1.0, -1.0, 0.0]) / math.sqrt(2)
    v = np.array([0.0, 0.0, 1.0])
    for i in range(4):
        a = 2 * math.pi * (i / 4 + t / 4)
        arm = math.cos(a) * u + math.sin(a) * v
        side = math.cos(a + math.pi / 2) * u + math.sin(a + math.pi / 2) * v
        tip = hub + d * 2.0 + arm * r
        shapes.append(S.rod(tuple(hub + d * 2.0), tuple(tip), 0.6, wood))
        p0, p1 = hub + d * 2.0 + arm * r * 0.25, tip
        shapes.append(hull([p0, p1, p0 + side * 6.0, p1 + side * 6.0, p0 + d * 0.3, p1 + side * 6.0 + d * 0.3],
                           canvas))
    return Part("sails", boxes=shapes)


def mill(style: str, age: int, t: float = 0.0, part: str = "all") -> Part:
    """The mill (2x2): a windmill tower in the culture's walls, its tattered sails turning; sacks and a pumpkin pile."""
    k = _k(style, age)
    out = [box((-26, -26, 0), (22, 22, 3.0), k.base)]
    if k.key == "F":
        out += H.walls(k, -18, -18, 14, 14, 3.0, 40) + H.pagoda_roof(k, -18, 14, -18, 14, 43.0, 14.0, over=6.0)
        hub = (16.0, 16.0, 34.0)
    elif k.key in ("M", "X"):
        out += [S.rod((-2.0, -2.0, 3.0), (-2.0, -2.0, 46.0), 17.0, k.wall, 14.0)]
        out += H.dome(-2.0, -2.0, 46.0, 14.0, k.roof) if k.key == "M" else \
            [S.rod((-2.0, -2.0, 46.0), (-2.0, -2.0, 60.0), 17.0, H.thatch(P.hexrgb("#7a6a40"), seed=2), 0.5)]
        hub = (10.0, 10.0, 40.0)
    else:
        prof = [(0, 0), (18.0, 0), (15.0, 44.0), (0, 44.0)]
        out += [S.turned((-2.0, -2.0, 3.0), prof, k.wall)]
        out += [S.turned((-2.0, -2.0, 47.0), [(0, 0), (17.0, 0), (12.0, 8.0), (4.0, 14.0), (0.1, 16.0)], k.roof)]
        hub = (10.0, 10.0, 46.0)
        for z in (14.0, 30.0):
            out.append(box((12.5, -4.0, z), (13.5, 0.0, z + 7.0), S.mat(k.glow, glow=1.0)))
    out.append(box((6.0, 8.0, 3.0), (16.0, 18.0, 16.0), H.facade(k.wall, [H.Opening("+y", 8.0, 14.0, 3.0, 13.0, "door")],
                                                                       k)))
    sack = H.plaster(P.hexrgb("#8a7a5a"), seed=4)
    out += [S.ball((20.0, 22.0, 2.5), (2.6, 2.2, 2.6), sack), S.ball((24.0, 18.0, 2.5), (2.4, 2.2, 2.4), sack)]
    out += H.pumpkins(-18.0, 26.0, 0.0, 3, seed=2)
    root = _part("mill", out)
    if part != "body":
        root.add(_sails(k, hub, t))
    return root


def blacksmith(style: str, age: int, t: float = 0.0, part: str = "all") -> Part:
    """The blacksmith (3x3): a smithy with a great chimney, its forge burning green, an anvil and racks of blades out
    front; ghostly green smoke rising from the chimney (animated)."""
    k = _k(style, age)
    out, top = _civic(k, -46, -46, 22, 18, 30, door_face="+x", axis="y")
    chim = P.bricks(P.hexrgb("#4a3a36"), P.hexrgb("#1e1612"), 2.6, 1.4, seed=8)
    cx, cy = -36.0, -8.0
    out += [box((cx - 6, cy - 6, 0), (cx + 6, cy + 6, top + 12), chim)]
    out += [box((24.0, -30.0, 0), (46.0, 20.0, 1.0), P.mottled(P.hexrgb("#3e3630"), P.hexrgb("#1a1612")))]  # yard
    forge = box((28.0, 4.0, 1.0), (38.0, 14.0, 7.0), chim)
    coals = S.ball((33.0, 9.0, 7.0), (4.0, 4.0, 1.2), S.Material(color=lambda l: P.mix(
        P.hexrgb("#7cff6a"), P.hexrgb("#1a6a2a"), P.noise(l.p, 0.8, 3)), glow=1.0, vivid=1.0))
    anvil = [box((33.0, -14.0, 1.0), (36.0, -9.0, 5.0), P.metal()), hull([(31.0, -15.0, 5.0), (38.0, -15.0, 5.0),
                                                                          (31.0, -8.0, 5.0), (38.0, -8.0, 5.0),
                                                                          (40.0, -11.5, 7.0), (31.0, -15.0, 7.0),
                                                                          (38.0, -15.0, 7.0), (31.0, -8.0, 7.0),
                                                                          (38.0, -8.0, 7.0)], P.metal())]
    out += [forge, coals, *anvil]
    rack_wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    out += [box((25.0, -28.0, 1.0), (27.0, -18.0, 12.0), rack_wood)]
    for i in range(3):
        out.append(S.rod((27.5, -26.0 + i * 3.5, 2.0), (28.0, -26.0 + i * 3.5, 14.0), 0.5, P.metal(rust=0.5), 0.05))
    out += H.wall_banner(23.0, -30.0, 28.0, "+x", 5.0, 12.0)
    root = _part("blacksmith", out)
    if part != "body":
        smoke = []
        for i in range(4):
            ph = (t + i / 4) % 1.0
            size = 4.0 + ph * 6.0
            c = (cx - 2 + ph * 6, cy + 2 - ph * 4, top + 14 + ph * 26)
            puff = S.Ellipsoid(S.Material(color=lambda l: np.broadcast_to(P.hexrgb("#8ad89a"), (len(l.p), 3)),
                                          glow=0.35 * (1 - ph) + 0.1, vivid=0.8), S._orient(c),
                               radii=(size, size, size * 0.8))
            puff.casts = False
            puff.ink = False
            smoke.append(puff)
        root.add(Part("smoke", boxes=smoke))
    return root


def university(style: str, age: int, t: float = 0.0, part: str = "all") -> Part:
    """The university (3x3): a necromancer's academy with an observatory tower; on top, a brass orrery of skulls
    turning round a green crystal (animated)."""
    k = _k(style, age)
    out, top = _civic(k, -50, -50, 26, 10, 36, door_face="+y", towers=[(14.0, -38.0, 10.0, 58.0)], seed=4)
    out += [box((-30.0, 12.0, 0.0), (26.0, 46.0, 1.0), P.bricks(P.hexrgb("#5a5660"), P.hexrgb("#2a2830"), 4.0, 4.0,
                                                                   moss=0.8))]
    out += H.wall_banner(27.0, -20.0, 32.0, "+x", 5.0, 14.0) + H.wall_banner(-12.0, 11.0, 32.0, "+y", 5.0, 14.0)
    # a lectern with a glowing book, and an urn of scrolls
    out += [box((4.0, 22.0, 1.0), (8.0, 26.0, 8.0), k.trim),
            box((3.0, 21.0, 8.0), (9.0, 27.0, 9.0), S.mat(P.hexrgb("#d8e8c0"), glow=0.6))]
    root = _part("university", out)
    if part != "body":
        c = np.array([-14.0, -20.0, top + 10.0])
        brass = P.metal(P.hexrgb("#a8823a"), shine=0.9)
        orb = S.ball(tuple(c), 3.0, S.mat(P.SLIME, glow=1.0))
        orb.casts = False
        rings = [orb]
        from .graveyard import skull
        for i, r in enumerate((7.0, 11.0)):
            a = 2 * math.pi * (t * (1 if i == 0 else -1) + i * 0.3)
            rings.append(S.torus(tuple(c), r, 0.3, brass, rot=(70, 0, math.degrees(a))))
            p = c + np.array([math.cos(a) * r, math.sin(a) * r * 0.35, math.sin(a) * r * 0.95])
            rings += skull(tuple(p - (0, 0, 1.2)), 1.0, i)
        rings.append(S.rod(tuple(c - (0, 0, 10.0)), tuple(c - (0, 0, 3.0)), 0.5, brass))
        root.add(Part("orrery", boxes=rings))
    return root


# --------------------------------------------------------------------------- camps

def lumber_camp(style: str, age: int = 2) -> Part:
    """The lumber camp (2x2): an open shed on posts, a pile of dead logs, a stump with an axe in it."""
    k = _k(style, age)
    out = [box((-26, -26, 0), (24, 22, 1.0), P.mottled(P.hexrgb("#3a3028"), P.hexrgb("#1a1410")))]
    for x, y in ((-24.0, -24.0), (6.0, -24.0), (-24.0, 4.0), (6.0, 4.0)):
        out.append(S.rod((x, y, 0.0), (x, y, 22.0), 1.0, k.trim))
    roof, _ = H.roof_for(k, -24.0, 6.0, -24.0, 4.0, 22.0)
    out += roof
    bark = P.bark(seed=2)
    for i in range(4):
        for j in range(3 - i // 2):
            z = 2.0 + i * 2.6
            out.append(S.rod((-20.0 + j * 0.5, -18.0 + i * 0.6 + j * 5.2, z), (2.0, -18.0 + i * 0.6 + j * 5.2, z), 1.3,
                             bark))
    out += [S.rod((14.0, 14.0, 0.0), (14.0, 14.0, 5.0), 4.0, bark),
            S.ball((14.0, 14.0, 5.0), (4.0, 4.0, 0.4), P.wood(P.hexrgb("#8a6a4a"), P.hexrgb("#4a3420"), along=0))]
    out += [S.rod((14.0, 14.0, 5.0), (15.0, 18.0, 13.0), 0.45, k.trim),
            hull([(13.8, 13.4, 5.0), (14.2, 13.4, 5.0), (13.8, 11.0, 6.5), (14.2, 11.0, 6.5), (13.8, 15.4, 7.8),
                  (14.2, 15.4, 7.8)], P.metal(rust=0.5))]
    out += H.team_flag((22.0, -20.0, 0.0), 22.0, 5) + H.door_light(6.0, 5.0, 16.0, k)
    return _part("lumber_camp", out)


def mining_camp(style: str, age: int = 2) -> Part:
    """The mining camp (2x2): a shed, a cart heaped with ore, pickaxes in a barrel, a lantern, a skull on a post."""
    k = _k(style, age)
    out = [box((-26, -26, 0), (24, 22, 1.0), P.mottled(P.hexrgb("#4a4440"), P.hexrgb("#201c1a")))]
    cab, top = _cottage(k, -26, -24, 4, 2, 22, "+x", "y")
    out += cab
    iron = P.metal(P.hexrgb("#3a363e"), rust=0.5)
    out += [box((10.0, -16.0, 3.0), (20.0, -2.0, 9.0), P.wood(P.hexrgb("#4a3020"), P.hexrgb("#1e120a"), along=1))]
    for y in (-14.0, -4.0):
        out.append(S.torus((9.6, y, 3.0), 2.4, 0.5, iron, rot=(0, 90, 0)))
    ore = [S.ball((15.0 + dx, -9.0 + dy, 10.0), r, P.mottled(P.hexrgb("#6a6670"), P.hexrgb("#2e2c34"), 1.0))
           for dx, dy, r in ((-2, -3, 2.6), (2, 1, 2.2), (-1, 3, 2.4))]
    out += ore + [S.ball((14.0, -9.0, 12.0), 1.0, P.metal(P.GOLD, shine=0.9))]
    out += [S.rod((16.0, 12.0, 0.0), (16.0, 12.0, 6.0), 2.4, P.wood(P.hexrgb("#4a3020"), P.hexrgb("#1e120a"), along=2))]
    for a in (-0.4, 0.3):
        out.append(S.rod((16.0, 12.0, 4.0), (16.0 + a * 6, 12.0 + a * 3, 13.0), 0.35, k.trim))
    from .graveyard import skull
    out += [S.rod((22.0, 18.0, 0.0), (22.0, 18.0, 10.0), 0.5, k.trim)] + skull((22.0, 18.0, 9.6), 1.6, 3)
    out += H.door_light(5.0, -12.0, 16.0, k) + H.team_flag((-24.0, 18.0, 0.0), 22.0, 6)
    return _part("mining_camp", out)


# --------------------------------------------------------------------------- military

def _scarecrow(x, y, seed=0) -> list:
    """A training dummy: a cross of poles in rags with a jack-o'-lantern head."""
    wood = P.wood(P.hexrgb("#4a3020"), P.hexrgb("#1e120a"), along=2)
    rag = S.Material(color=lambda l: P.mix(P.hexrgb("#5a4a3a"), P.hexrgb("#2a2018"), P.noise(l.p, 1.0, seed)))
    return [S.rod((x, y, 0.0), (x, y, 16.0), 0.6, wood), S.rod((x - 6.0, y, 12.0), (x + 6.0, y, 12.0), 0.5, wood),
            S.rod((x, y, 14.0), (x, y, 6.0), 3.0, rag, 4.0)] + PK.lantern((x, y, 15.0), 2.6, seed=seed)


def barracks(style: str, age: int) -> Part:
    """The barracks (3x3): a hall, a training yard in front with pumpkin-headed dummies and a rack of weapons."""
    k = _k(style, age)
    out, top = _civic(k, -52, -52, 18, 4, 30, door_face="+y", towers=[(-50.0, 10.0, 7.0, 44.0)] if k.key == "W" else
                      [], seed=7)
    out += [box((-40.0, 6.0, 0.0), (40.0, 52.0, 0.8), P.mottled(P.hexrgb("#3e3630"), P.hexrgb("#1a1612")))]
    out += _scarecrow(10.0, 30.0, 1) + _scarecrow(28.0, 18.0, 2)
    rack_wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    out += [box((22.0, 34.0, 0.0), (44.0, 36.0, 1.0), rack_wood), box((22.0, 34.0, 10.0), (44.0, 36.0, 11.0), rack_wood)]
    for i in range(5):
        x = 24.0 + i * 4.5
        out.append(S.rod((x, 35.0, 0.5), (x, 35.0, 16.0 + (i % 2) * 2), 0.35, rack_wood))
        out.append(S.hull([(x - 0.7, 35.0, 16.0 + (i % 2) * 2), (x + 0.7, 35.0, 16.0 + (i % 2) * 2),
                           (x, 35.0, 19.0 + (i % 2) * 2)], P.metal(rust=0.5)))
    out += H.wall_banner(19.0, -30.0, 28.0, "+x", 5.0, 12.0) + H.wall_banner(-30.0, 5.0, 28.0, "+y", 5.0, 12.0)
    out += H.team_flag((44.0, 48.0, 0.0), 26.0, 7)
    return _part("barracks", out)


def archery_range(style: str, age: int) -> Part:
    """The archery range (3x3): a long open shed and a line of targets, straw boards painted with skulls, shot full
    of arrows."""
    k = _k(style, age)
    out = [box((-56, -56, 0), (56, 56, 0.8), P.mottled(P.hexrgb("#3a3428"), P.hexrgb("#1a1610")))]
    out += H.walls(k, -54, -54, -14, 20, 2.0, 26, door=("+x", -20.0, 8.0, 14.0))
    roof, top = H.roof_for(k, -54, -14, -54, 20, 28.0, axis="y")
    out += roof
    for i, (x, y) in enumerate(((10.0, -30.0), (26.0, -6.0), (42.0, 18.0))):
        def face(look, i=i):
            p = look.p
            r = np.hypot(p[:, 0], p[:, 2])
            ring = (np.floor(r / 1.4) % 2 == 0)
            c = np.where(ring[:, None], P.hexrgb("#c8b878"), P.hexrgb("#8a2a1a"))
            skull = r < 1.2
            return np.where(skull[:, None], P.hexrgb("#e8e0d0"), c)
        t = S.Material(color=face, vivid=0.6)
        out += [S.rod((x, y, 0.0), (x, y, 10.0), 0.5, k.trim), S.rod((x, y - 0.6, 12.0), (x, y + 0.6, 12.0), 5.0, t)]
        for a in range(3):
            ang = a * 2.1 + i
            out.append(S.rod((x + math.cos(ang) * 2.0, y + 0.6, 12.0 + math.sin(ang) * 2.0),
                             (x + math.cos(ang) * 2.0 + 0.5, y + 6.0, 12.6 + math.sin(ang) * 2.0), 0.15, k.trim))
    out += H.wall_banner(-13.0, -30.0, 24.0, "+x", 5.0, 12.0) + H.team_flag((-14.0, 26.0, 0.0), 24.0, 8)
    return _part("archery_range", out)


def stable(style: str, age: int) -> Part:
    """The stable (3x3): a long barn, a paddock fenced in iron, hay, a trough of black water and a horse skull."""
    k = _k(style, age)
    out, top = _civic(k, -54, -54, 4, 10, 26, door_face="+x", axis="y")
    out += [box((6.0, -50.0, 0.0), (54.0, 50.0, 0.8), P.mottled(P.hexrgb("#3e3428"), P.hexrgb("#1a1410")))]
    out += H.fence((10.0, 50.0), (54.0, 50.0), 7.0) + H.fence((54.0, -50.0), (54.0, 50.0), 7.0)
    hay = H.thatch(P.hexrgb("#8a7a48"), seed=5)
    out += [S.ball((24.0, -30.0, 3.0), (6.0, 5.0, 4.0), hay), S.ball((34.0, -22.0, 2.4), (4.0, 4.0, 3.0), hay)]
    out += [box((20.0, 20.0, 0.0), (40.0, 26.0, 5.0), k.trim),
            box((21.0, 21.0, 4.0), (39.0, 25.0, 4.6), S.mat(P.hexrgb("#141a18"), shine=0.9, gloss=60))]
    from .graveyard import skull
    out += [S.rod((6.0, 14.0, 6.0), (6.0, 14.0, 28.0), 0.5, k.trim)]
    out += [S.Ellipsoid(P.bone(3), S._orient((6.6, 14.0, 24.0)), radii=(1.4, 1.8, 3.0))]
    out += H.wall_banner(5.0, -30.0, 22.0, "+x", 5.0, 12.0) + H.team_flag((50.0, -46.0, 0.0), 24.0, 9)
    return _part("stable", out)


def siege_workshop(style: str, age: int = 3) -> Part:
    """The siege workshop (4x4): a big barn with wide doors, a half-built bone catapult and stacks of coffins."""
    k = _k(style, age)
    out, top = _civic(k, -80, -80, 20, 20, 40, door_face="+y", axis="x")
    out += [box((-70.0, 22.0, 0.0), (70.0, 80.0, 0.8), P.mottled(P.hexrgb("#3e3428"), P.hexrgb("#1a1410")))]
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    bone = P.bone(5)
    out += [box((24.0, 30.0, 4.0), (50.0, 50.0, 7.0), wood)]
    for x in (28.0, 46.0):
        for y in (30.0, 50.0):
            out.append(S.torus((x, y + (1.0 if y > 40 else -1.0), 4.0), 3.6, 0.5, P.metal(), rot=(0, 90, 0)))
    out += [S.rod((37.0, 40.0, 7.0), (37.0, 52.0, 26.0), 1.2, bone, 0.8), S.ball((37.0, 52.0, 26.0), 2.0, bone)]
    for i in range(3):
        out.append(hull([(x, y, z) for x, y in ((-50.0 + i * 2, 40.0 + i * 3), (-30.0 + i * 2, 40.0 + i * 3),
                                                (-28.0 + i * 2, 46.0 + i * 3), (-52.0 + i * 2, 46.0 + i * 3))
                         for z in (i * 4.0, i * 4.0 + 4.0)], P.wood(P.hexrgb("#2a1e18"), P.hexrgb("#100a08"), along=0)))
    out += H.wall_banner(21.0, -40.0, 36.0, "+x", 6.0, 14.0) + H.team_flag((64.0, 70.0, 0.0), 28.0, 10)
    return _part("siege_workshop", out)


# --------------------------------------------------------------------------- market, monastery, dock

def market(style: str, age: int) -> Part:
    """The market (4x4): a trading hall at the back and three stalls under awnings in the owner's colour, selling
    pumpkins, potions and skulls."""
    k = _k(style, age)
    out, top = _civic(k, -80, -80, 0, -10, 32, door_face="+y", axis="x")
    out += [box((-80.0, -8.0, 0.0), (80.0, 80.0, 0.8), P.bricks(P.hexrgb("#5a5660"), P.hexrgb("#2a2830"), 5.0, 5.0,
                                                                   moss=0.9))]
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)

    def stripes(look):
        return np.where((look.p[:, 0] + look.p[:, 1]) % 6.0 < 3.0, 0.75, 0.42)
    for i, (x, y) in enumerate(((-40.0, 30.0), (10.0, 40.0), (50.0, 0.0))):
        out += [box((x - 12, y - 6, 0.0), (x + 12, y + 6, 8.0), wood)]
        for dx in (-11.0, 11.0):
            out.append(S.rod((x + dx, y + 6.0, 0.0), (x + dx, y + 6.0, 22.0), 0.6, wood))
            out.append(S.rod((x + dx, y - 6.0, 0.0), (x + dx, y - 6.0, 24.0), 0.6, wood))
        out.append(hull([(x - 13, y - 7, 24.0), (x + 13, y - 7, 24.0), (x - 13, y + 9, 20.0), (x + 13, y + 9, 20.0),
                         (x - 13, y - 7, 24.8), (x + 13, y - 7, 24.8), (x - 13, y + 9, 20.8), (x + 13, y + 9, 20.8)],
                        P.team(1.0, stripes)))
        if i == 0:
            for j in range(4):
                out += PK.lantern((x - 8 + j * 5.0, y + 1.0, 8.0), 2.0, seed=j)
        elif i == 1:
            for j in range(6):
                col = [P.hexrgb("#7cff6a"), P.hexrgb("#c040ff"), P.hexrgb("#ff3a40")][j % 3]
                bottle = S.ball((x - 9 + j * 3.6, y + 2.0, 9.5), (1.0, 1.0, 1.4), S.mat(col, glow=0.7, vivid=1.0))
                bottle.casts = False
                out += [bottle, S.rod((x - 9 + j * 3.6, y + 2.0, 10.6), (x - 9 + j * 3.6, y + 2.0, 12.0), 0.35,
                                      S.mat(P.hexrgb("#6a5040")))]
        else:
            from .graveyard import skull
            for j in range(3):
                out += skull((x - 7 + j * 6.0, y + 2.0, 8.0), 1.6, j)
    out += H.team_flag((70.0, 70.0, 0.0), 30.0, 11) + H.wall_banner(1.0, -40.0, 30.0, "+x", 6.0, 14.0)
    return _part("market", out)


def monastery(style: str, age: int = 3) -> Part:
    """The monastery (3x3): the culture's house of the dead (W a gothic chapel, E a black church with a needle
    spire, M a domed mausoleum, F a temple with a torii gate, X an ossuary chapel), in a graveyard."""
    k = _k(style, age)
    out, top = _civic(k, -50, -50, 14, 8, 36, door_face="+y", axis="y",
                      towers=[(-46.0, 16.0, 8.0, 60.0)] if k.key in ("W", "E", "G") else [], seed=12)
    if k.key == "E":
        out += H.spire(-18.0, -21.0, top - 8.0, 5.0, 30.0, k.roof)
    out += [box((-50.0, 10.0, 0.0), (56.0, 56.0, 0.6), P.mottled(P.hexrgb("#2e3628"), P.hexrgb("#141810")))]
    for i, (x, y) in enumerate(((24.0, 24.0), (36.0, 36.0), (20.0, 44.0), (44.0, 18.0), (34.0, 4.0), (48.0, 46.0))):
        out += H.gravestone(x, y, 0.0, 6.0 + (i % 2) * 2, i, seed=i)
    out += H.fence((54.0, -40.0), (54.0, 54.0), 7.0) + H.fence((-40.0, 54.0), (54.0, 54.0), 7.0)
    if k.key == "F":  # a torii gate
        red = S.mat(P.hexrgb("#8a1a14"), shine=0.4, vivid=0.7)
        out += [S.rod((-4.0, 30.0, 0.0), (-4.0, 30.0, 26.0), 1.2, red), S.rod((18.0, 30.0, 0.0), (18.0, 30.0, 26.0), 1.2,
                                                                                red),
                box((-10.0, 29.0, 26.0), (24.0, 31.0, 28.5), S.mat(P.hexrgb("#1a1414"))),
                box((-7.0, 29.4, 21.0), (21.0, 30.6, 22.5), red)]
    elif k.key == "X":
        out += H.marigolds(8.0, 22.0, 0.0, 10) + H.candles(-10.0, 22.0, 0.0, 6) + H.candles(4.0, 28.0, 0.0, 5)
    elif k.key == "M":
        out += H.sarcophagus(16.0, -20.0, 0.0, "x") + H.urn(20.0, 0.0, 0.0, 6.0)
    else:
        out += H.cauldron(-20.0, 22.0, 0.0, 3.0)
    out += H.wall_banner(15.0, -20.0, 32.0, "+x", 5.0, 14.0)
    return _part("monastery", out)


def dock(style: str, age: int) -> Part:
    """The dock (3x3, on the water): a pier of black planks on posts, a boathouse, a crane with a cage, lanterns."""
    k = _k(style, age)
    deck_m = P.wood(P.hexrgb("#3a2e26"), P.hexrgb("#14100c"), along=0, planks=2.2, seed=4)
    out = [box((-58.0, -58.0, 4.0), (58.0, 30.0, 6.0), deck_m)]
    for x in np.linspace(-56, 56, 6):
        for y in (-56.0, 28.0):
            out.append(S.rod((x, y, -6.0), (x, y, 8.0), 1.4, P.wood(P.hexrgb("#2a2018"), P.hexrgb("#0e0a08"), along=2)))
    cab, top = _cottage(k, -54, -54, -6, -6, 26, "+y", "x", z0=6.0)
    out += cab
    crane_m = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=2)
    out += [S.rod((30.0, 10.0, 6.0), (30.0, 10.0, 44.0), 1.2, crane_m), S.rod((30.0, 10.0, 42.0), (50.0, 26.0, 46.0),
                                                                             0.9, crane_m)]
    out += [S.rod((50.0, 26.0, 46.0), (50.0, 26.0, 30.0), 0.15, S.mat(P.hexrgb("#8a7a5a")))]
    iron = P.metal(P.hexrgb("#2c2a32"), rust=0.5)
    for a in range(6):  # a hanging cage
        ang = 2 * math.pi * a / 6
        out.append(S.rod((50.0 + math.cos(ang) * 2.4, 26.0 + math.sin(ang) * 2.4, 22.0),
                         (50.0 + math.cos(ang) * 2.4, 26.0 + math.sin(ang) * 2.4, 30.0), 0.18, iron))
    out += [S.rod((50.0, 26.0, 21.6), (50.0, 26.0, 22.2), 2.6, iron), S.rod((50.0, 26.0, 29.8), (50.0, 26.0, 30.4), 2.6,
                                                                           iron)]
    from .ghostships import lantern_post
    out += lantern_post((56.0, 28.0, 6.0), 8.0) + lantern_post((-56.0, 28.0, 6.0), 8.0)
    out += H.team_flag((0.0, 28.0, 6.0), 26.0, 12)
    return _part("dock", out)


# --------------------------------------------------------------------------- fields, sites, ruins

def farm(stage: float = 1.0) -> Part:
    """A pumpkin patch (3x3): furrows of dark earth, vines, pumpkins growing with `stage` (0 just dug, 1 ripe) and a
    scarecrow."""
    soil = S.Material(color=lambda l: P.mix(P.hexrgb("#3a2a1e"), P.hexrgb("#1a120c"),
                                            0.5 + 0.5 * np.sin(l.p[:, 0] * 0.6)))
    out = [box((-60.0, -60.0, 0.0), (60.0, 60.0, 1.0), soil)]
    rng = np.random.default_rng(3)
    vine = P.mottled(P.hexrgb("#3d5a22"), P.hexrgb("#1a2a10"), 1.0, 0.4)
    for row in range(5):
        x = -48.0 + row * 24.0
        if stage > 0.05:
            out.append(S.rod((x, -56.0, 1.0), (x, 56.0, 1.0), 1.0 + 1.2 * stage, vine))
        for j in range(5):
            y = -48.0 + j * 24.0 + rng.uniform(-4, 4)
            r = 1.0 + 3.6 * stage * rng.uniform(0.7, 1.1)
            if stage > 0.15:
                out.append(S.Ellipsoid(P.pumpkin(rot=0.2, seed=row * 5 + j), S._orient((x + rng.uniform(-3, 3), y, r * 0.8)),
                                       radii=(r * 1.1, r, r * 0.8)))
    out += _scarecrow(0.0, 0.0, 9)
    return _part("farm", out)


def fish_trap(stage: float = 1.0) -> Part:
    """A fish trap: a ring of black stakes in the water, a sagging net, bony fish caught in it (more with `stage`)."""
    wood = P.wood(P.hexrgb("#2a2018"), P.hexrgb("#0e0a08"), along=2)
    out = []
    for i in range(14):
        a = 2 * math.pi * i / 14
        out.append(S.rod((math.cos(a) * 40.0, math.sin(a) * 40.0, -4.0), (math.cos(a) * 40.0, math.sin(a) * 40.0, 10.0),
                         1.2, wood, 0.6))
    net = S.Material(color=lambda l: np.where((np.abs(np.sin(l.p[:, 0] * 1.2)) < 0.25)[:, None] |
                                              (np.abs(np.sin(l.p[:, 1] * 1.2)) < 0.25)[:, None],
                                              P.hexrgb("#6a5a44"), P.hexrgb("#1e2a28")))
    out.append(S.torus((0, 0, 7.0), 40.0, 0.4, S.mat(P.hexrgb("#6a5a44"))))
    rng = np.random.default_rng(5)
    for i in range(int(round(stage * 6))):
        x, y = rng.uniform(-24, 24, 2)
        out.append(S.Ellipsoid(P.bone(i), S._orient((x, y, 1.0), (0, 0, rng.uniform(0, 180))), radii=(1.2, 3.4, 1.0)))
    return _part("fish_trap", out)


def construction(tiles: float, variant: int = 0, progress: float = 0.5) -> Part:
    """A building site: dug earth, a frame of black scaffold poles rising with `variant`, planks, a lantern."""
    size = FOOT.get(int(round(tiles)), tiles * 45.0) * 0.45
    prog = min(1.0, 0.25 + 0.25 * (variant % 4))
    dirt = P.mottled(P.hexrgb("#3a2e24"), P.hexrgb("#1a140e"), 2.0, 0.5)
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=2)
    out = [box((-size, -size, 0.0), (size, size, 0.8), dirt)]
    h = 10.0 + 30.0 * prog
    for x in (-size + 2, size - 2):
        for y in (-size + 2, size - 2):
            out.append(S.rod((x, y, 0.0), (x, y, h), 0.6, wood))
    for z in np.arange(8.0, h, 10.0):
        out.append(S.rod((-size + 2, size - 2, z), (size - 2, size - 2, z), 0.45, wood))
        out.append(S.rod((size - 2, -size + 2, z), (size - 2, size - 2, z), 0.45, wood))
    out += [box((-size * 0.4, -size * 0.2, 0.8), (size * 0.3, size * 0.1, 3.0), wood)]
    out += H.door_light(size - 2.0, size - 4.0, 8.0, H.kit("G", 1))
    return _part("site", out)


def rubble(tiles: float, variant: int = 0) -> Part:
    """What is left of a building: broken stones, charred beams, a skull, embers."""
    size = FOOT.get(int(round(tiles)), tiles * 45.0) * 0.45
    rng = np.random.default_rng(17 + variant)
    stone = P.mottled(P.hexrgb("#5a5660"), P.hexrgb("#24222a"), 1.5, 0.5)
    char = S.mat(P.hexrgb("#141012"))
    out = [box((-size, -size, 0.0), (size, size, 0.6), P.mottled(P.hexrgb("#2a2420"), P.hexrgb("#100c0a")))]
    for i in range(int(6 + tiles * 4)):
        c = np.array([rng.uniform(-size, size), rng.uniform(-size, size), 0.0])
        r = rng.uniform(2.0, 4.5)
        out.append(hull(c + rng.normal(0, r * 0.6, (6, 3)) * np.array([1, 1, 0.6]) + (0, 0, r * 0.4), stone))
    for i in range(3):
        a = np.array([rng.uniform(-size, size), rng.uniform(-size, size), 0.5])
        out.append(S.rod(tuple(a), tuple(a + (rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(1, 6))), 1.0,
                         char))
    from .graveyard import skull
    out += skull((rng.uniform(-size / 2, size / 2), rng.uniform(-size / 2, size / 2), 0.6), 1.6, variant)
    return _part("rubble", out)
