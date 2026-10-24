"""Every economic and military building, as haunted houses (spooky.py) in the culture kits (haunted.py), as Parts
centred on their footprints (2x2 = 80 units, 3x3 = 128, 4x4 = 176). Each is a scary house first (the player asked:
crooked steep roofs, boarded windows, broken shutters, porches, turrets, spider webs, ravens, graves; no pyramids),
and says what it is by what stands round it (a barracks has a training yard with scarecrow dummies, a market its
stalls, a stable its paddock); its culture says how it is built.

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
from . import spooky as Y
from .geometry import Part
from .haunted import FOOT, box, hull


def _k(style: str, age: int) -> H.Kit:
    return H.kit("G" if style == "G" else style, age if style != "G" else 1)


def _part(name, shapes, *children) -> Part:
    return Part(name, boxes=list(shapes)).add(*children)


def _culture_props(k: H.Kit, x, y, seed=0) -> list:
    """What lies about a house in each culture: pumpkins and a cauldron, gravestones, urns, paper lanterns,
    marigolds and candles."""
    if k.key in ("G", "W"):
        return H.pumpkins(x, y, 0.0, 2, seed=seed)
    if k.key == "E":
        return Y.graves([(x, y), (x + 6.0, y - 5.0)], seed)
    if k.key == "M":
        return H.urn(x, y, 0.0, 5.0) + H.urn(x + 5.0, y - 3.0, 0.0, 4.0)
    if k.key == "F":
        return H.lanterns_hanging([(x, y, 12.0)], k) + [S.rod((x, y, 0.0), (x, y, 16.0), 0.4, k.trim)]
    return H.marigolds(x, y, 0.0, 6, seed=seed) + H.candles(x + 5.0, y - 2.0, 0.0, 3, seed=seed)


def papel_picado(a, b, seed=0) -> list:
    """A string of cut-paper flags in bright colours sagging between two points (Day of the Dead)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    cols = [P.hexrgb("#e0407a"), P.hexrgb("#f08a10"), P.hexrgb("#8a40d0"), P.hexrgb("#20b0b0"), P.hexrgb("#f0d020")]
    out = []
    n = max(3, int(np.linalg.norm(b - a) // 3.0))
    pts = [a + (b - a) * i / n - np.array([0, 0, 2.5 * math.sin(math.pi * i / n)]) for i in range(n + 1)]
    for p, q in zip(pts, pts[1:]):
        out.append(S.rod(tuple(p), tuple(q), 0.1, S.mat(P.hexrgb("#e8e0d0"))))
    for i, p in enumerate(pts[1:-1]):
        d = (b - a) / np.linalg.norm(b - a) * 0.9
        flag = hull([p - d, p + d, p - d - (0, 0, 2.2), p + d - (0, 0, 2.2), p - (0, 0, 2.8)],
                    S.mat(cols[(i + seed) % len(cols)], vivid=1.0))
        flag.casts = False
        out.append(flag)
    return out


# --------------------------------------------------------------------------- house

def house(style: str, age: int, variant: int = 0) -> Part:
    """A haunted home on a 2x2 footprint, in three variants (the game's three house pictures)."""
    k = _k(style, age)
    v = variant % 3
    seed = 100 + v
    parts = []
    if k.key == "G":  # a crooked witch's cottage under a steep, sagging thatch, a leaning chimney
        out, top = Y.manor(k, -24, -20, 14, 12, 1, "x" if v != 1 else "y", "+y" if v != 1 else "+x",
                           roof_kind="gable", porch_at=False, dormers=0, chimney_at=(-14.0, -6.0) if v != 2 else
                           (6.0, -10.0), lean=3.0, seed=seed, storey_h=15.0)
        out += H.door_light(-14.0 if v != 1 else 15.0, 13.5 if v != 1 else -10.0, 13.0, k)
        out += H.cauldron(22.0, 20.0, 0.0, 3.0, seed=v) + H.pumpkins(4.0, 24.0, 0.0, 2, seed=v)
        out += Y.fence((-30.0, 32.0, 0.0), (10.0, 32.0, 0.0), 5.0, seed) if v == 0 else []
        if v == 2:
            parts.append(Y.dead_tree(24.0, -14.0, 46.0, v))
    elif k.key == "W":
        if v == 0:  # two storeys, a porch, a witch-hat turret on the corner
            out, top = Y.manor(k, -26, -24, 12, 12, 2, "x", "+y", door_at=-8.0, tower=(22.0, 22.0, 6.5, 44.0),
                               dormers=1, chimney_at=(-18.0, -12.0), seed=seed)
        elif v == 1:  # the Second Empire house: a mansard roof with dormers and cresting
            out, top = Y.manor(k, -26, -26, 14, 12, 2, "x", "+x", roof_kind="mansard", seed=seed,
                               chimney_at=(-18.0, -16.0))
        else:  # a tall gable end facing the street, a round attic window, a lean-to porch
            out, top = Y.manor(k, -22, -26, 14, 14, 2, "y", "+y", door_at=-6.0, dormers=1, chimney_at=(-12.0, -18.0),
                               seed=seed)
        out += _culture_props(k, 26.0, 30.0, v) + H.wall_banner(13.0 if v != 1 else 15.5, -10.0, 24.0, "+x", 4.0, 9.0)
    elif k.key == "E":
        if v == 0:  # a narrow gothic house, its steep gable end to the street, a needle spire
            out, top = Y.manor(k, -22, -26, 12, 14, 2, "y", "+y", dormers=0, seed=seed)
            out += H.spire(-5.0, -18.0, top - 10.0, 2.2, 16.0, k.roof)
        elif v == 1:  # a manor with a square tower and a spire
            out, top = Y.manor(k, -26, -24, 12, 12, 2, "x", "+y", door_at=-10.0, tower=(20.0, 20.0, 6.0, 46.0),
                               dormers=1, seed=seed)
        else:  # a long stone house, a chimney, gargoyles
            out, top = Y.manor(k, -26, -22, 14, 12, 2, "x", "+x", dormers=2, chimney_at=(-18.0, -8.0), seed=seed)
        out += _culture_props(k, 26.0, 30.0, v) + H.wall_banner(13.0 if v else 13.0, -12.0, 26.0, "+x", 4.0, 10.0)
    elif k.key == "M":
        if v == 0:  # two storeys behind a crumbling parapet, a minaret with an onion dome
            out, top = Y.manor(k, -26, -24, 12, 12, 2, "x", "+y", door_at=-8.0, tower=(20.0, 20.0, 5.0, 46.0),
                               seed=seed)
        elif v == 1:  # a cracked house and a dome over its upper room
            out, top = Y.manor(k, -26, -24, 14, 12, 2, "x", "+x", seed=seed)
            out += H.dome(-6.0, -6.0, top, 9.0, k.roof, onion=True)
        else:  # a low house and a wind tower
            out, top = Y.manor(k, -26, -24, 14, 12, 1, "x", "+y", seed=seed, storey_h=16.0)
            out += Y.storey(k, -26, -24, -14, -12, top - 2.6, 22.0, m=k.upper, every=6.0, seed=seed, boarded=0.5)
            out += Y.roof(k, -26, -14, -24, -12, top + 19.4, "flat", seed=seed)[0]
        out += _culture_props(k, 24.0, 30.0, v) + H.wall_banner(13.0, -10.0, 24.0, "+x", 4.0, 10.0)
    elif k.key == "F":
        if v == 0:  # a dark house under a curved roof, torn paper walls glowing
            out, top = Y.manor(k, -26, -22, 14, 12, 1, "x", "+y", seed=seed, storey_h=18.0)
        elif v == 1:  # two storeys and a lantern tower
            out, top = Y.manor(k, -26, -24, 10, 12, 2, "x", "+y", door_at=-10.0, tower=(20.0, 20.0, 6.0, 34.0),
                               seed=seed)
        else:
            out, top = Y.manor(k, -22, -26, 14, 12, 2, "y", "+x", seed=seed)
        out += _culture_props(k, 26.0, 26.0, v) + H.wall_banner(15.0, -10.0, 24.0, "+x", 4.0, 10.0)
    else:  # X: a painted house, a tiled roof or a parapet, bunting, marigolds and candles
        if v == 0:
            out, top = Y.manor(k, -26, -24, 12, 12, 2, "x", "+y", door_at=-8.0, dormers=0, seed=seed,
                               roof_kind="tiles" if k.age >= 3 else "gable")
        elif v == 1:  # a house and a painted belfry
            out, top = Y.manor(k, -26, -24, 10, 12, 1, "x", "+y", door_at=-10.0, tower=(20.0, 20.0, 5.0, 30.0),
                               seed=seed, storey_h=17.0, roof_kind="tiles" if k.age >= 3 else "gable")
        else:
            out, top = Y.manor(k, -22, -24, 14, 12, 2, "y", "+x", roof_kind="flat", seed=seed)
        out += papel_picado((-24.0, 26.0, 14.0), (20.0, 26.0, 14.0), v)
        out += [S.rod((-24.0, 26.0, 0.0), (-24.0, 26.0, 15.0), 0.4, k.trim), S.rod((20.0, 26.0, 0.0), (20.0, 26.0, 15.0),
                                                                                 0.4, k.trim)]
        out += _culture_props(k, 26.0, 30.0, v) + PK.lantern((-18.0, 30.0, 0.0), 2.2, seed=v)
    return _part("house", out, *parts)


# --------------------------------------------------------------------------- the big halls

def _hall(k: H.Kit, x0, y0, x1, y1, floors=2, door_face="+y", towers=(), axis="x", seed=0, roof_kind=None,
          chimney_at=None, dormers=2, storey_h=16.0) -> tuple[list, float]:
    """A large haunted house, with turrets at the given (x, y, r, h) spots."""
    out, top = Y.manor(k, x0, y0, x1, y1, floors, axis, door_face, roof_kind=roof_kind, dormers=dormers,
                       chimney_at=chimney_at, seed=seed, storey_h=storey_h, tower=towers[0] if towers else None)
    for (tx, ty, tr, th) in towers[1:]:
        out += Y.turret(k, tx, ty, tr, 0.0, th, seed=seed + 1)
    return out, top


def town_center(style: str, age: int) -> Part:
    """The Town Center (4x4): the village's great haunted house, three storeys, a turret, a porch, flags, and a
    graveyard by the gate."""
    k = _k(style, age)
    floors = 2 if k.key == "G" else 3
    out, top = _hall(k, -62, -60, 26, 20, floors, "+y", towers=[(-58.0, 40.0, 11.0, 74.0 if floors == 3 else 60.0)],
                     chimney_at=(-48.0, -40.0), dormers=3, seed=1, axis="x",
                     roof_kind="mansard" if k.key == "W" and k.age >= 3 else None)
    out += H.wall_banner(27.0 + (3.2 if k.key in ("W", "G") else 0.0), -30.0, 40.0, "+x", 6.0, 16.0)
    out += H.wall_banner(27.0 + (3.2 if k.key in ("W", "G") else 0.0), 0.0, 40.0, "+x", 6.0, 16.0)
    out += H.team_flag((40.0, 40.0, 0.0), 30.0, 2)
    out += Y.graves([(52.0, 46.0), (62.0, 34.0), (46.0, 60.0), (60.0, 56.0)], 3)
    out += Y.fence((34.0, 70.0, 0.0), (74.0, 70.0, 0.0), 6.0, 4) + Y.fence((74.0, 24.0, 0.0), (74.0, 70.0, 0.0), 6.0, 5)
    out += _culture_props(k, 40.0, 10.0, 1) + H.cauldron(50.0, -10.0, 0.0, 4.0)
    if k.key == "X":
        out += papel_picado((30.0, 30.0, 18.0), (30.0, -50.0, 18.0), 2)
    parts = [Y.dead_tree(66.0, -40.0, 64.0, 2)]
    return _part("town_center", out, *parts)


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
    """The mill (2x2): a crooked windmill tower in the culture's walls, boarded windows, its tattered sails turning;
    sacks, a pumpkin pile and a web on the door."""
    k = _k(style, age)
    out = [box((-26, -26, 0), (22, 22, 3.0), k.base)]
    if k.key == "F":
        out += Y.storey(k, -18, -18, 14, 14, 3.0, 40, seed=3)
        out += Y.roof(k, -18, 14, -18, 14, 43.0, "curved", "x", over=5.0, seed=3)[0]
        hub = (16.0, 16.0, 34.0)
    elif k.key in ("M", "X"):
        out += [S.rod((-2.0, -2.0, 3.0), (-2.0, -2.0, 46.0), 17.0, k.wall, 14.0)]
        out += H.dome(-2.0, -2.0, 46.0, 14.0, k.roof, onion=True) if k.key == "M" else \
            [S.rod((-2.0, -2.0, 46.0), (-2.0, -2.0, 62.0), 17.0, H.thatch(P.hexrgb("#7a6a40"), seed=2), 0.5)]
        hub = (10.0, 10.0, 40.0)
    else:
        prof = [(0, 0), (18.0, 0), (15.0, 44.0), (0, 44.0)]
        out += Y.tilt([S.turned((-2.0, -2.0, 3.0), prof, k.wall),
                       S.turned((-2.0, -2.0, 47.0), [(0, 0), (17.5, 0), (13.0, 9.0), (5.0, 16.0), (0.1, 19.0)],
                                Y.holey(k.roof, 5))], (-2.0, -2.0, 3.0), (2.0, -1.5, 0.0))
        hub = (11.0, 10.0, 46.0)
        for z in (14.0, 30.0):
            out.append(box((12.5, -4.0, z), (13.5, 0.0, z + 7.0), S.mat(k.glow, glow=1.0)))
            out += [S.rod((13.8, -4.6, z + 0.6), (13.8, 0.6, z + 6.4), 0.6, Y.DARK_WOOD),
                    S.rod((13.8, -4.6, z + 6.4), (13.8, 0.6, z + 0.6), 0.6, Y.DARK_WOOD)]
    out.append(box((6.0, 8.0, 3.0), (16.0, 18.0, 16.0), H.facade(k.wall, [H.Opening("+y", 8.0, 14.0, 3.0, 13.0, "door")],
                                                                       k)))
    out += Y.web((16.2, 18.2, 15.8), (-1.0, 0, 0), (0, 0, -1.0), 5.0, 7)
    sack = H.plaster(P.hexrgb("#8a7a5a"), seed=4)
    out += [S.ball((20.0, 22.0, 2.5), (2.6, 2.2, 2.6), sack), S.ball((24.0, 18.0, 2.5), (2.4, 2.2, 2.4), sack)]
    out += H.pumpkins(-18.0, 26.0, 0.0, 3, seed=2) + Y.ravens([(-2.0, -2.0, 66.0 if k.key not in ("F", "M") else 60.0)])
    root = _part("mill", out)
    if part != "body":
        root.add(_sails(k, hub, t))
    return root


def blacksmith(style: str, age: int, t: float = 0.0, part: str = "all") -> Part:
    """The blacksmith (3x3): a sooty haunted smithy with a great leaning chimney, its forge burning green, an anvil
    and racks of blades out front; ghostly smoke rising from the chimney (animated)."""
    k = _k(style, age)
    out, top = _hall(k, -50, -46, 14, 14, 1, "+x", axis="y", seed=6, dormers=1, storey_h=20.0)
    chim = P.bricks(P.hexrgb("#4a3a36"), P.hexrgb("#1e1612"), 2.6, 1.4, moss=0.3, seed=8)
    cx, cy = -38.0, -6.0
    out += Y.tilt([box((cx - 6, cy - 6, 0), (cx + 6, cy + 6, top + 12), chim),
                   box((cx - 7, cy - 7, top + 12), (cx + 7, cy + 7, top + 14), chim)], (cx, cy, 0.0), (2.5, -2.0, 0.0))
    out += [box((18.0, -30.0, 0), (50.0, 24.0, 1.0), P.mottled(P.hexrgb("#3e3630"), P.hexrgb("#1a1612")))]  # yard
    forge = box((30.0, 4.0, 1.0), (40.0, 14.0, 7.0), chim)
    coals = S.ball((35.0, 9.0, 7.0), (4.0, 4.0, 1.2), S.Material(color=lambda l: P.mix(
        P.hexrgb("#7cff6a"), P.hexrgb("#1a6a2a"), P.noise(l.p, 0.8, 3)), glow=1.0, vivid=1.0))
    anvil = [box((35.0, -14.0, 1.0), (38.0, -9.0, 5.0), P.metal()), hull([(33.0, -15.0, 5.0), (40.0, -15.0, 5.0),
                                                                          (33.0, -8.0, 5.0), (40.0, -8.0, 5.0),
                                                                          (42.0, -11.5, 7.0), (33.0, -15.0, 7.0),
                                                                          (40.0, -15.0, 7.0), (33.0, -8.0, 7.0),
                                                                          (40.0, -8.0, 7.0)], P.metal())]
    out += [forge, coals, *anvil]
    rack_wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    out += [box((44.0, -28.0, 1.0), (46.0, -18.0, 12.0), rack_wood)]
    for i in range(3):
        out.append(S.rod((46.5, -26.0 + i * 3.5, 2.0), (47.0, -26.0 + i * 3.5, 14.0), 0.5, P.metal(rust=0.6), 0.05))
    out += H.wall_banner(15.0, -30.0, 22.0, "+x", 5.0, 10.0) + Y.web((15.0, 14.5, 21.0), (0, -1.0, 0), (0, 0, -1.0),
                                                                        7.0, 61)
    root = _part("blacksmith", out)
    if part != "body":
        smoke = []
        for i in range(4):
            ph = (t + i / 4) % 1.0
            size = 4.0 + ph * 6.0
            c = (cx + 1 + ph * 6, cy + 2 - ph * 4, top + 16 + ph * 26)
            puff = S.Ellipsoid(S.Material(color=lambda l: np.broadcast_to(P.hexrgb("#8ad89a"), (len(l.p), 3)),
                                          glow=0.35 * (1 - ph) + 0.1, vivid=0.8), S._orient(c),
                               radii=(size, size, size * 0.8))
            puff.casts = False
            puff.ink = False
            smoke.append(puff)
        root.add(Part("smoke", boxes=smoke))
    return root


def university(style: str, age: int, t: float = 0.0, part: str = "all") -> Part:
    """The university (3x3): a necromancer's academy, three storeys under a steep roof, an observatory turret; on
    top, a brass orrery of skulls turning round a glowing crystal (animated)."""
    k = _k(style, age)
    out, top = _hall(k, -52, -50, 12, 6, 3, "+y", towers=[(18.0, -36.0, 9.0, 64.0)], seed=4, dormers=2,
                     chimney_at=(-40.0, -38.0), storey_h=14.0)
    out += [box((-40.0, 18.0, 0.0), (26.0, 50.0, 1.0), P.bricks(P.hexrgb("#5a5660"), P.hexrgb("#2a2830"), 4.0, 4.0,
                                                                   moss=0.8))]
    out += H.wall_banner(13.0 + (3.2 if k.key in ("W", "G") else 0), -24.0, 36.0, "+x", 5.0, 14.0)
    out += [box((30.0, 26.0, 1.0), (34.0, 30.0, 8.0), k.trim),
            box((29.0, 25.0, 8.0), (35.0, 31.0, 9.0), S.mat(P.hexrgb("#d8e8c0"), glow=0.6))]
    out += Y.graves([(40.0, 44.0), (48.0, 36.0)], 4)
    root = _part("university", out)
    if part != "body":
        c = np.array([-20.0, -22.0, top + 12.0])
        brass = P.metal(P.hexrgb("#a8823a"), shine=0.9)
        orb = S.ball(tuple(c), 3.0, S.mat(P.hexrgb("#c060ff"), glow=1.0))
        orb.casts = False
        rings = [orb]
        from .graveyard import skull
        for i, r in enumerate((7.0, 11.0)):
            a = 2 * math.pi * (t * (1 if i == 0 else -1) + i * 0.3)
            rings.append(S.torus(tuple(c), r, 0.3, brass, rot=(70, 0, math.degrees(a))))
            p = c + np.array([math.cos(a) * r, math.sin(a) * r * 0.35, math.sin(a) * r * 0.95])
            rings += skull(tuple(p - (0, 0, 1.2)), 1.0, i)
        rings.append(S.rod(tuple(c - (0, 0, 12.0)), tuple(c - (0, 0, 3.0)), 0.5, brass))
        root.add(Part("orrery", boxes=rings))
    return root


# --------------------------------------------------------------------------- camps

def lumber_camp(style: str, age: int = 2) -> Part:
    """The lumber camp (2x2): a crooked shack, a pile of dead logs, a stump with an axe in it, a web."""
    k = _k(style, age)
    out = [box((-26, -26, 0), (24, 22, 1.0), P.mottled(P.hexrgb("#3a3028"), P.hexrgb("#1a1410")))]
    cab, top = Y.manor(k, -26, -26, 0, -2, 1, "y", "+x", porch_at=False, dormers=0, seed=21, storey_h=13.0, lean=3.0)
    out += cab
    bark = P.bark(seed=2)
    for i in range(4):
        for j in range(3 - i // 2):
            z = 2.0 + i * 2.6
            out.append(S.rod((4.0 + j * 0.5, -22.0 + i * 0.6 + j * 5.2, z), (20.0, -22.0 + i * 0.6 + j * 5.2, z), 1.3,
                             bark))
    out += [S.rod((10.0, 14.0, 0.0), (10.0, 14.0, 5.0), 4.0, bark),
            S.ball((10.0, 14.0, 5.0), (4.0, 4.0, 0.4), P.wood(P.hexrgb("#8a6a4a"), P.hexrgb("#4a3420"), along=0))]
    out += [S.rod((10.0, 14.0, 5.0), (11.0, 18.0, 13.0), 0.45, k.trim),
            hull([(9.8, 13.4, 5.0), (10.2, 13.4, 5.0), (9.8, 11.0, 6.5), (10.2, 11.0, 6.5), (9.8, 15.4, 7.8),
                  (10.2, 15.4, 7.8)], P.metal(rust=0.6))]
    out += H.team_flag((-22.0, 14.0, 0.0), 22.0, 5) + Y.ravens([(20.0, -10.0, 10.5)], 2)
    return _part("lumber_camp", out, Y.dead_tree(-14.0, 14.0, 40.0, 3))


def mining_camp(style: str, age: int = 2) -> Part:
    """The mining camp (2x2): a crooked shack, a cart heaped with ore, pickaxes in a barrel, a skull on a post."""
    k = _k(style, age)
    out = [box((-26, -26, 0), (24, 22, 1.0), P.mottled(P.hexrgb("#4a4440"), P.hexrgb("#201c1a")))]
    cab, top = Y.manor(k, -26, -24, 2, 2, 1, "y", "+x", porch_at=False, dormers=0, seed=22, storey_h=13.0, lean=3.0,
                       chimney_at=(-18.0, -16.0))
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
    out += H.team_flag((-24.0, 18.0, 0.0), 22.0, 6) + Y.web((2.4, 2.4, 14.0), (0, -1.0, 0), (0, 0, -1.0), 5.0, 22)
    return _part("mining_camp", out)


# --------------------------------------------------------------------------- military

def _scarecrow(x, y, seed=0) -> list:
    """A training dummy: a cross of poles in rags with a jack-o'-lantern head."""
    wood = P.wood(P.hexrgb("#4a3020"), P.hexrgb("#1e120a"), along=2)
    rag = S.Material(color=lambda l: P.mix(P.hexrgb("#5a4a3a"), P.hexrgb("#2a2018"), P.noise(l.p, 1.0, seed)))
    return [S.rod((x, y, 0.0), (x, y, 16.0), 0.6, wood), S.rod((x - 6.0, y, 12.0), (x + 6.0, y, 12.0), 0.5, wood),
            S.rod((x, y, 14.0), (x, y, 6.0), 3.0, rag, 4.0)] + PK.lantern((x, y, 15.0), 2.6, seed=seed)


def barracks(style: str, age: int) -> Part:
    """The barracks (3x3): a haunted manor, a training yard in front with pumpkin-headed dummies and a rack of
    weapons."""
    k = _k(style, age)
    out, top = _hall(k, -56, -56, 8, -2, 2, "+y", towers=[(-50.0, 8.0, 7.0, 46.0)] if k.key in ("W", "E") else [],
                     seed=7, chimney_at=(-44.0, -44.0), dormers=2, storey_h=15.0)
    out += [box((-40.0, 18.0, 0.0), (40.0, 56.0, 0.8), P.mottled(P.hexrgb("#3e3630"), P.hexrgb("#1a1612")))]
    out += _scarecrow(10.0, 34.0, 1) + _scarecrow(28.0, 22.0, 2) + _scarecrow(-14.0, 44.0, 3)
    rack_wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    out += [box((26.0, 40.0, 0.0), (48.0, 42.0, 1.0), rack_wood), box((26.0, 40.0, 10.0), (48.0, 42.0, 11.0), rack_wood)]
    for i in range(5):
        x = 28.0 + i * 4.5
        out.append(S.rod((x, 41.0, 0.5), (x, 41.0, 16.0 + (i % 2) * 2), 0.35, rack_wood))
        out.append(S.hull([(x - 0.7, 41.0, 16.0 + (i % 2) * 2), (x + 0.7, 41.0, 16.0 + (i % 2) * 2),
                           (x, 41.0, 19.0 + (i % 2) * 2)], P.metal(rust=0.6)))
    out += Y.fence((-50.0, 58.0, 0.0), (20.0, 58.0, 0.0), 6.0, 7) + Y.fence((50.0, 10.0, 0.0), (50.0, 50.0, 0.0), 6.0, 8)
    out += H.wall_banner(9.0 + (1.6 if k.key in ("W", "G") else 0), -30.0, 26.0, "+x", 5.0, 12.0)
    out += H.team_flag((48.0, 52.0, 0.0), 26.0, 7)
    return _part("barracks", out)


def archery_range(style: str, age: int) -> Part:
    """The archery range (3x3): a long, crooked haunted hall and a line of targets, straw boards painted with
    skulls, shot full of arrows, a scarecrow keeping score."""
    k = _k(style, age)
    out = [box((-60, -60, 0), (60, 60, 0.8), P.mottled(P.hexrgb("#3a3428"), P.hexrgb("#1a1610")))]
    hall, top = _hall(k, -58, -58, -20, 26, 1, "+x", axis="y", seed=8, dormers=2, storey_h=18.0,
                      roof_kind="gambrel" if k.key in ("G", "W") else None)
    out += hall
    for i, (x, y) in enumerate(((8.0, -36.0), (24.0, -10.0), (40.0, 16.0))):
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
    out += _scarecrow(46.0, 40.0, 4) + Y.ravens([(40.0, 16.0, 17.2)], 8)
    out += H.wall_banner(-19.0, -36.0, 24.0, "+x", 5.0, 12.0) + H.team_flag((-16.0, 34.0, 0.0), 24.0, 8)
    return _part("archery_range", out)


def stable(style: str, age: int) -> Part:
    """The stable (3x3): a sagging barn under a two-pitched roof, a paddock fenced with broken pickets, hay, a
    trough of black water and a horse's skull on a post."""
    k = _k(style, age)
    out, top = _hall(k, -58, -56, 2, 12, 1, "+x", axis="y", seed=9, roof_kind="gambrel", dormers=0, storey_h=22.0)
    out += [box((8.0, -50.0, 0.0), (56.0, 54.0, 0.8), P.mottled(P.hexrgb("#3e3428"), P.hexrgb("#1a1410")))]
    out += Y.fence((12.0, 54.0, 0.0), (56.0, 54.0, 0.0), 7.0, 9) + Y.fence((56.0, -50.0, 0.0), (56.0, 54.0, 0.0), 7.0,
                                                                           10)
    hay = H.thatch(P.hexrgb("#8a7a48"), seed=5)
    out += [S.ball((26.0, -34.0, 3.0), (6.0, 5.0, 4.0), hay), S.ball((36.0, -26.0, 2.4), (4.0, 4.0, 3.0), hay)]
    out += [box((22.0, 22.0, 0.0), (42.0, 28.0, 5.0), k.trim),
            box((23.0, 23.0, 4.0), (41.0, 27.0, 4.6), S.mat(P.hexrgb("#141a18"), shine=0.9, gloss=60))]
    out += [S.rod((48.0, 40.0, 0.0), (48.0, 40.0, 14.0), 0.5, k.trim),
            S.Ellipsoid(P.bone(3), S._orient((48.6, 40.0, 12.0)), radii=(1.4, 1.8, 3.0))]
    out += Y.web((56.0, 30.0, 6.8), (0, 1.0, 0), (0, 0, -1.0), 5.0, 9) + Y.ravens([(56.0, -20.0, 7.2)], 9)
    out += H.wall_banner(3.0, -34.0, 22.0, "+x", 5.0, 12.0) + H.team_flag((52.0, -46.0, 0.0), 24.0, 9)
    return _part("stable", out, Y.dead_tree(24.0, 44.0, 46.0, 1))


def siege_workshop(style: str, age: int = 3) -> Part:
    """The siege workshop (4x4): a huge, leaning barn with a lit hay door, a half-built bone catapult and stacks of
    coffins."""
    k = _k(style, age)
    out, top = _hall(k, -84, -80, 20, 20, 1, "+y", axis="x", seed=10, roof_kind="gambrel", dormers=0, storey_h=30.0,
                     chimney_at=(-70.0, -60.0))
    out += [box((-70.0, 30.0, 0.0), (74.0, 84.0, 0.8), P.mottled(P.hexrgb("#3e3428"), P.hexrgb("#1a1410")))]
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)
    bone = P.bone(5)
    out += [box((28.0, 36.0, 4.0), (54.0, 56.0, 7.0), wood)]
    for x in (32.0, 50.0):
        for y in (36.0, 56.0):
            out.append(S.torus((x, y + (1.0 if y > 46 else -1.0), 4.0), 3.6, 0.5, P.metal(rust=0.6), rot=(0, 90, 0)))
    out += [S.rod((41.0, 46.0, 7.0), (41.0, 58.0, 26.0), 1.2, bone, 0.8), S.ball((41.0, 58.0, 26.0), 2.0, bone)]
    for i in range(3):
        out.append(hull([(x, y, z) for x, y in ((-50.0 + i * 2, 46.0 + i * 3), (-30.0 + i * 2, 46.0 + i * 3),
                                                (-28.0 + i * 2, 52.0 + i * 3), (-52.0 + i * 2, 52.0 + i * 3))
                         for z in (i * 4.0, i * 4.0 + 4.0)], P.wood(P.hexrgb("#2a1e18"), P.hexrgb("#100a08"), along=0)))
    out += Y.fence((70.0, 30.0, 0.0), (70.0, 80.0, 0.0), 6.0, 11)
    out += H.wall_banner(21.0, -40.0, 34.0, "+x", 6.0, 14.0) + H.team_flag((64.0, 74.0, 0.0), 28.0, 10)
    return _part("siege_workshop", out, Y.dead_tree(60.0, -50.0, 56.0, 4))


# --------------------------------------------------------------------------- market, monastery, dock

def market(style: str, age: int) -> Part:
    """The market (4x4): a haunted trading house at the back and three stalls under awnings in the owner's
    colour, selling pumpkins, potions and skulls."""
    k = _k(style, age)
    out, top = _hall(k, -84, -84, 4, -16, 2, "+y", axis="x", seed=11, dormers=3, chimney_at=(-70.0, -74.0),
                     towers=[(14.0, -24.0, 7.0, 46.0)] if k.key in ("W", "X", "F") else [])
    out += [box((-80.0, 0.0, 0.0), (80.0, 80.0, 0.8), P.bricks(P.hexrgb("#5a5660"), P.hexrgb("#2a2830"), 5.0, 5.0,
                                                                  moss=0.9))]
    wood = P.wood(P.hexrgb("#3a2a20"), P.hexrgb("#160e0a"), along=1)

    def stripes(look):
        return np.where((look.p[:, 0] + look.p[:, 1]) % 6.0 < 3.0, 0.75, 0.42)
    for i, (x, y) in enumerate(((-40.0, 40.0), (10.0, 52.0), (54.0, 10.0))):
        out += [box((x - 12, y - 6, 0.0), (x + 12, y + 6, 8.0), wood)]
        for dx in (-11.0, 11.0):
            out.append(S.rod((x + dx, y + 6.0, 0.0), (x + dx, y + 6.0, 22.0), 0.6, wood))
            out.append(S.rod((x + dx, y - 6.0, 0.0), (x + dx, y - 6.0, 24.0), 0.6, wood))
        out.append(hull([(x - 13, y - 7, 24.0), (x + 13, y - 7, 24.0), (x - 13, y + 9, 20.0), (x + 13, y + 9, 20.0),
                         (x - 13, y - 7, 24.8), (x + 13, y - 7, 24.8), (x - 13, y + 9, 20.8), (x + 13, y + 9, 20.8)],
                        P.team(1.0, stripes)))
        out += Y.web((x + 11.0, y + 6.0, 21.0), (-1.0, 0, 0), (0, 0, -1.0), 5.0, 30 + i)
        if i == 0:
            for j in range(4):
                out += PK.lantern((x - 8 + j * 5.0, y + 1.0, 8.0), 2.0, seed=j)
        elif i == 1:
            for j in range(6):
                col = [P.hexrgb("#ff8a20"), P.hexrgb("#c040ff"), P.hexrgb("#ff3a40")][j % 3]
                bottle = S.ball((x - 9 + j * 3.6, y + 2.0, 9.5), (1.0, 1.0, 1.4), S.mat(col, glow=0.7, vivid=1.0))
                bottle.casts = False
                out += [bottle, S.rod((x - 9 + j * 3.6, y + 2.0, 10.6), (x - 9 + j * 3.6, y + 2.0, 12.0), 0.35,
                                      S.mat(P.hexrgb("#6a5040")))]
        else:
            from .graveyard import skull
            for j in range(3):
                out += skull((x - 7 + j * 6.0, y + 2.0, 8.0), 1.6, j)
    if k.key == "X":
        out += papel_picado((-60.0, 20.0, 26.0), (60.0, 20.0, 26.0), 1)
    out += H.team_flag((70.0, 70.0, 0.0), 30.0, 11) + Y.ravens([(-28.0, 46.0, 8.4)], 11)
    return _part("market", out)


def monastery(style: str, age: int = 3) -> Part:
    """The monastery (3x3): the culture's haunted chapel under a steep roof and a tall tower (a witch-hat turret, a
    needle spire, a minaret, a lantern tower, a painted belfry), in a graveyard of leaning stones and a broken fence."""
    k = _k(style, age)
    out, top = _hall(k, -54, -54, 4, 6, 1, "+y", axis="y", seed=12, dormers=0, storey_h=28.0,
                     towers=[(-48.0, 14.0, 8.0, 70.0)], roof_kind="gable" if k.key in ("G", "W", "E") else None)
    if k.key == "E":
        out += H.spire(-25.0, -20.0, top - 10.0, 4.0, 26.0, k.roof)
    out += [box((-50.0, 16.0, 0.0), (58.0, 58.0, 0.6), P.mottled(P.hexrgb("#2e3628"), P.hexrgb("#141810")))]
    out += Y.graves([(24.0, 26.0), (36.0, 38.0), (20.0, 46.0), (46.0, 20.0), (34.0, 4.0), (50.0, 48.0), (10.0, 34.0)],
                    12)
    out += Y.fence((56.0, -46.0, 0.0), (56.0, 56.0, 0.0), 7.0, 12) + Y.fence((-44.0, 56.0, 0.0), (56.0, 56.0, 0.0),
                                                                             7.0, 13)
    if k.key == "F":  # a torii gate
        red = S.mat(P.hexrgb("#8a1a14"), shine=0.4, vivid=0.7)
        out += [S.rod((-4.0, 34.0, 0.0), (-4.0, 34.0, 26.0), 1.2, red), S.rod((18.0, 34.0, 0.0), (18.0, 34.0, 26.0), 1.2,
                                                                                red),
                box((-10.0, 33.0, 26.0), (24.0, 35.0, 28.5), S.mat(P.hexrgb("#1a1414"))),
                box((-7.0, 33.4, 21.0), (21.0, 34.6, 22.5), red)]
    elif k.key == "X":
        out += H.marigolds(8.0, 26.0, 0.0, 10) + H.candles(-10.0, 26.0, 0.0, 6) + H.candles(4.0, 32.0, 0.0, 5)
        out += papel_picado((-40.0, 20.0, 20.0), (40.0, 20.0, 20.0), 3)
    elif k.key == "M":
        out += H.sarcophagus(18.0, -20.0, 0.0, "x") + H.urn(22.0, 0.0, 0.0, 6.0)
    else:
        out += H.cauldron(-24.0, 28.0, 0.0, 3.0)
    out += H.wall_banner(5.0, -24.0, 30.0, "+x", 5.0, 14.0)
    return _part("monastery", out, Y.dead_tree(44.0, -30.0, 58.0, 0))


def dock(style: str, age: int) -> Part:
    """The dock (3x3, on the water): a pier of black planks on posts, a crooked boathouse, a crane with a cage,
    lanterns and a web."""
    k = _k(style, age)
    deck_m = P.wood(P.hexrgb("#3a2e26"), P.hexrgb("#14100c"), along=0, planks=2.2, seed=4)
    out = [box((-58.0, -58.0, 4.0), (58.0, 30.0, 6.0), deck_m)]
    for x in np.linspace(-56, 56, 6):
        for y in (-56.0, 28.0):
            out.append(S.rod((x, y, -6.0), (x, y, 8.0), 1.4, P.wood(P.hexrgb("#2a2018"), P.hexrgb("#0e0a08"), along=2)))
    cab, top = Y.manor(k, -54, -54, -8, -8, 1, "x", "+y", porch_at=False, dormers=1, seed=13, storey_h=18.0, z0=4.0)
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
    from .graveyard import skull
    out += skull((50.0, 26.0, 22.4), 1.4, 5)
    from .ghostships import lantern_post
    out += lantern_post((56.0, 28.0, 6.0), 8.0) + lantern_post((-56.0, 28.0, 6.0), 8.0)
    out += H.team_flag((0.0, 28.0, 6.0), 26.0, 12) + Y.web((30.0, 10.0, 40.0), (0.7, 0.7, 0), (0, 0, -1.0), 6.0, 13)
    out += Y.ravens([(30.0, 10.0, 44.4)], 12)
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
