"""The giant Pac-Man as a walking haunted castle (the player asked, 2026-10-24: "make it look like a Halloween castle,
and when it moves the castle is showing legs when walking").

A castle of black stone torn out of the ground, a lump of rock and dangling roots under it: a keep with lit gothic
windows under a steep crooked roof, four corner towers under bent witch's hats with red pennants, battlemented
curtain walls, and on the front a gatehouse that is its face: two lit windows for eyes over a glowing arched gate,
its portcullis a row of iron teeth (he still bites with Pac-Man's bite). Bats round the towers, webs, a raven.

Six spider legs of black chitin with glowing joints carry it. At rest (the "walker" rig, animation.py) the castle
sits on its rock and the legs are folded away out of sight; walking, it rises on them and strides, three legs at a
time. He is red whoever owns him, as the giant always was: his pennants are in the player colour, which the build
draws in the palette's red team shades.
"""
from __future__ import annotations

import math

import numpy as np

from . import haunted as H
from . import paint as P
from . import sculpt as S
from . import spooky as Y
from .geometry import Part
from .haunted import box, hull
from .units import Unit

WALKER_SCALE = 4.0  # screen size: at rest a little bigger than a Castle, striding on its legs bigger still
LIFT = 30.0  # how high the castle's base rides on its legs
ROCK = 9.0  # the lump of rock under it: at rest the castle sits on it
LEGS = 6
HIP_R, KNEE_R, FOOT_R = 20.0, 36.0, 50.0  # how far from the middle the legs leave the rock, bend, and reach the ground


def _kit() -> H.Kit:
    k = H.kit("E", 4)
    return H.Kit(**{**k.__dict__, "roof": H.kit("W", 4).roof, "glow": P.hexrgb("#ff9a2a"), "crest": True})


def _leg(k: int) -> Part:
    """A spider's leg: from the rock out and up to a knee, then down to a claw on the ground; black chitin, joints
    glowing orange."""
    a = math.radians(30.0 + 60.0 * k)
    d = np.array([math.cos(a), math.sin(a), 0.0])
    hip = d * HIP_R + (0, 0, LIFT - 4.0)
    knee = d * KNEE_R + (0, 0, LIFT + 26.0)  # spider's knees rise high above the body
    ankle = d * (FOOT_R - 3.0) + (0, 0, 12.0)
    foot = d * FOOT_R
    chitin = S.Material(color=lambda l: P.mix(P.hexrgb("#2a1e2e"), P.hexrgb("#08060a"), 0.5 * P.fbm(l.p, 2.0, k)),
                        shine=0.8, gloss=40, vivid=0.3)
    joint = S.mat(P.hexrgb("#ff7a1a"), glow=0.9, vivid=1.0)
    shapes = [S.rod(tuple(hip), tuple(knee), 3.2, chitin, 2.4), S.ball(tuple(knee), 2.8, joint),
              S.rod(tuple(knee), tuple(ankle), 2.4, chitin, 1.5), S.ball(tuple(ankle), 1.7, joint),
              S.rod(tuple(ankle), tuple(foot), 1.5, chitin, 0.35), S.ball(tuple(hip), 3.4, joint)]
    for j in range(4):  # bristles along the upper leg
        p = hip + (knee - hip) * (0.25 + 0.18 * j)
        shapes.append(S.rod(tuple(p), tuple(p + (0, 0, 2.2) + d * 0.6), 0.25, chitin, 0.05))
    return Part(f"leg_{k}", pivot=tuple(hip), boxes=shapes)


def walking_castle(key: str = "giant_pacman") -> Unit:
    k = _kit()
    z0 = LIFT  # the castle's base, riding on the legs
    rng = np.random.default_rng(66)
    rock_m = P.mottled(P.hexrgb("#4a3e36"), P.hexrgb("#1a1410"), 2.0, 0.5, 7)
    pts = [(math.cos(a) * 26.0, math.sin(a) * 26.0, z0) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    pts += [(rng.uniform(-12, 12), rng.uniform(-12, 12), z0 - ROCK + rng.uniform(-1, 2)) for _ in range(8)]
    out = [hull(pts, rock_m)]
    root_m = P.bark(seed=4)
    for j in range(9):  # roots dangling from the rock
        a = 2 * math.pi * j / 9 + rng.uniform(-0.2, 0.2)
        p = np.array([math.cos(a) * 16.0, math.sin(a) * 16.0, z0 - 4.0])
        out += [S.rod(tuple(p), tuple(p + (math.cos(a) * 4.0, math.sin(a) * 4.0, -7.0 - rng.uniform(0, 5))), 0.8,
                      root_m, 0.1)]
    base = z0 + 0.5
    stone = k.wall
    # curtain walls between the towers, battlemented
    for (x0, y0, x1, y1) in ((-22, 19, 22, 23), (19, -22, 23, 22), (-22, -23, 22, -19), (-23, -22, -19, 22)):
        out.append(box((x0, y0, base), (x1, y1, base + 16.0), stone))
        n = int(max(x1 - x0, y1 - y0) // 4)
        for i in range(n):
            if i % 2:
                continue
            if x1 - x0 > y1 - y0:
                u = x0 + (x1 - x0) * i / n
                out.append(box((u, y0, base + 16.0), (u + (x1 - x0) / n, y1, base + 19.0), stone))
            else:
                u = y0 + (y1 - y0) * i / n
                out.append(box((x0, u, base + 16.0), (x1, u + (y1 - y0) / n, base + 19.0), stone))
    # the keep: two storeys with lit gothic windows (some boarded), a steep crooked roof with dormers
    out += Y.storey(k, -13, -13, 13, 13, base, 15.0, seed=1, boarded=0.2, shutters=0.0)
    out += Y.storey(k, -13, -13, 13, 13, base + 15.0, 13.0, m=k.upper, seed=2, boarded=0.25, shutters=0.0)
    roof, top = Y.roof(k, -13, 13, -13, 13, base + 28.0, "gable", "x", rise=22.0, over=2.5, seed=3)
    out += Y.tilt(roof, (0, 0, base + 28.0), (2.5, -2.0, 0.0))
    out += H.spire(0.0, 0.0, top - 4.0, 2.4, 16.0, k.roof)
    # four corner towers under bent witch's hats, red pennants
    for i, (x, y) in enumerate(((21.0, 21.0), (21.0, -21.0), (-21.0, 21.0), (-21.0, -21.0))):
        out.append(S.rod((x, y, base), (x, y, base + 30.0), 6.0, stone))
        out.append(S.rod((x, y, base + 30.0), (x, y, base + 32.0), 7.2, k.base))
        prof = [(0, 0), (7.6, 0), (6.8, 1.0), (4.4, 6.0), (2.4, 12.0), (0.9, 18.0), (0.1, 21.0)]
        out.append(S.turned((x, y, base + 32.0), prof, Y.holey(k.roof, i), rot=(6 * (1 if x > 0 else -1), -5, 0)))
        out += H.team_flag((x + (1.5 if x > 0 else -1.5), y, base + 51.0), 9.0, i)
        for z in (base + 12.0, base + 22.0):  # arrow slits lit from inside
            for fx, fy in ((x + 5.9, y), (x, y + 5.9)):
                out.append(box((fx - 0.4 if fx != x else fx - 0.8, fy - 0.8 if fy == y else fy - 0.4, z),
                               (fx + 0.4 if fx != x else fx + 0.8, fy + 0.8 if fy == y else fy + 0.4, z + 3.5),
                               S.mat(k.glow, glow=1.0)))
    # the gatehouse on the front: its face, two lit windows for eyes over a glowing gate with iron teeth
    eyes = [H.Opening("+y", -7.0, -2.5, base + 19.0, base + 25.0, "gothic"),
            H.Opening("+y", 2.5, 7.0, base + 19.0, base + 25.0, "gothic"),
            H.Opening("+y", -6.0, 6.0, base, base + 15.0, "round")]
    face_kit = H.Kit(**{**k.__dict__, "glow": P.hexrgb("#ff6a1a")})
    out.append(box((-10.0, 20.0, base), (10.0, 27.0, base + 28.0), H.facade(stone, eyes, face_kit)))
    out.append(box((-11.0, 19.5, base + 28.0), (11.0, 27.5, base + 30.0), k.base))
    for i in range(6):  # crenellations
        out.append(box((-11.0 + i * 4.0, 19.5, base + 30.0), (-9.0 + i * 4.0, 27.5, base + 33.0), stone))
    iron = P.metal(P.hexrgb("#2a262a"), rust=0.5, seed=3)
    for i in range(7):  # the portcullis: a row of iron teeth across the gate
        x = -5.4 + i * 1.8
        out.append(S.rod((x, 27.3, base + 13.5), (x, 27.3, base + 6.0 + (1.5 if i % 2 else 0.0)), 0.45, iron, 0.05))
    out.append(S.rod((-6.0, 27.3, base + 13.6), (6.0, 27.3, base + 13.6), 0.45, iron))
    for sx in (1, -1):  # angry brows over the eyes
        out.append(S.rod((sx * 2.0, 27.4, base + 26.6), (sx * 7.6, 27.4, base + 25.0), 0.6, k.trim))
    out += H.bats([(14.0, 30.0, base + 58.0), (-18.0, 10.0, base + 64.0), (6.0, -20.0, base + 70.0),
                   (28.0, -6.0, base + 50.0)], seed=6)
    out += Y.web((22.0, 23.2, base + 15.5), (-1.0, 0, 0), (0, 0, -1.0), 7.0, 6)
    out += Y.ravens([(0.0, 23.5, base + 33.2)], 6)
    castle = Part("castle", pivot=(0.0, 0.0, z0), boxes=out)
    root = Part("root").add(castle, *[_leg(i) for i in range(LEGS)])
    return Unit(key, "Giant Pac-Man", "Saboteur (cheat)", "walker", root, group="easter", attack="chomp",
                scale=WALKER_SCALE)
