"""Witch's cauldrons on the brew: Villagers build one on the sea of brew (terrain 15, lava.py) by their island's beach,
and it boils over at the enemies that come within its range, hurling bubbling brew at ships, units and buildings.
(Age of Minecraft's volcano, the same in the game's data: the code still calls it the volcano.)

The game has a tower made for the sea that it never lets anyone build: the Sea Tower (unit 785). It is still a tower
(unit class 52), so the tower upgrades reach it: Fletching, Bodkin Arrow and Bracer, Masonry and Architecture, and
Heated Shot against ships. `patch` makes it the volcano, in place, for every civilisation:

- Villagers build it (`BUTTON` on their military build page), from the start of the game, on the lava by a beach: it takes
  the Dock's rules (the beach next to it, the ground it may cover, how level), with the tile under its middle on the
  lava (its placement terrain, the rule that keeps a Dock on water). Fishing Ships cannot build it: the game lets a
  Fishing Ship build the Fish Trap alone (it checks for the Fish Trap's unit, 004b9b8f among others), and a test
  game with three more test volcanoes for Fishing Ships, on the Dock's rules, the Fish Trap's and its own, placed
  none of them.
- It is three tiles across, as big as a Barracks, built on the Barracks' building site, and it sinks like a Dock when
  destroyed.
- It shoots its own shots (786, and 787: the game left both unused, the Sea Tower shot the towers' arrow), which
  become lava bombs, lobbed high (`ARC`) from its crater: piercing damage for everything, more against ships and
  buildings (`ATTACK`).
- It has three pictures of its own, new SLPs written next to the others: the volcano at rest (its own graphic,
  STWR1NN), the eruption it plays each time it fires (787's graphic, MFSTW, made one picture of `ERUPTION_FRAMES`:
  its attack graphic, with the Trebuchet's firing boom), and the lava bomb (786's graphic, MRSTW, a tumbling glowing
  rock of `BOMB_FRAMES`). Both shot graphics drew the old arrows in 32 directions; a volcano and a ball of rock look
  the same from every side, so they become one direction. They also take the sequence types of the pictures they
  now are (`ERUPTING`, `TUMBLING`): an arrow's frames are not played over time, so with the arrows' type the eruption
  stood still on its first picture and the volcano, waiting for the picture its bomb leaves on (`FRAME`), never
  fired (seen in game, 2026-10-11).
- Its icon is added at the end of the four building icon sheets (one per age), and its texts name it the Volcano.
"""
from __future__ import annotations

import math
import struct
import zlib
from typing import Optional

import numpy as np

from . import datfile, slp
from . import datunits as DU
from . import voxel as V
from .geometry import Part, cuboid
from .units import Unit

SEA_TOWER = 785
SHOTS = (786, 787)  # the Sea Tower's own shots, which nothing shoots (it shot the towers' arrow): both lava bombs
LOOK = "STWR"  # the Sea Tower's graphics' names
VILLAGER = 118  # where buildings say Villagers build them
DOCK = "DOCK"  # the Dock's unit name: the volcano takes its placement rules
# in a Villager's build menu, on the page its interface kind picks (10, a tower's: the military buildings; 2 is
# the economic ones), where 5 is free: the Siege Workshop has 4, the Outpost 6 (a 14th button did not show)
BUTTON = 5
LAVA = 15  # the terrain (lava.LAVA)
COST = (2, 150, 1, 3, 100, 1, -1, 0, 0)  # 150 stone, 100 gold
TIME = 60  # seconds to build
HP = 1500
# (class, amount): pierce for everything, extra against ships (armour class 16) and buildings (11)
ATTACK = [(3, 12), (16, 18), (11, 15)]
ARMOUR = {4: 3, 3: 10}  # melee, pierce: its other armour classes (as a building and a tower) stay
RANGE = 8.0  # tiles, a tower's
SIGHT = 10.0
RELOAD = 3.0  # seconds between eruptions
ARC = 0.6  # how high the bombs lob: a tower's arrow 0.05, a Mangonel's stone 0.4
SIZE = 1.5  # tiles from its centre to its edge: three tiles across
FRAME = 4  # the eruption's picture its bomb leaves on
ERUPTION_FRAMES = 12
ERUPTION_SECONDS = 0.1  # a picture: the eruption plays 1.2 s of its 3 s reload
BOMB_FRAMES = 8
BOMB_SECONDS = 0.08
# sequence types (datfile.Graphic.sequence_at): the eruption plays like an attack that shoots on a frame (an
# archer's, the Trebuchet's: 3), the bomb tumbles like the Trebuchet's rock (1); the arrows' type (2) never plays
ERUPTING, TUMBLING, STILL = 3, 1, 0
BUILDING_LAYER, SHOT_LAYER = 20, 30
SITE, SINKING = "CNST3_NN", "DEXP3_NN"  # the Barracks' building site; the Dock's going down
BOOM = "trebfire.wav"  # the Trebuchet's firing boom, on each eruption
BUILDING_ICONS = (50705, 50706, 50707, 50708)  # in interfac.drs: one sheet per age
HELP_STRINGS = 79000  # as gameplay.HELP_STRINGS
TEXTS = {"name": "Cauldron", "creation": "Build Cauldron",
         "help": "Build <b> Cauldron<b> (<cost>) \nA witch's cauldron on the brew, by your island's beach. It boils "
                 "over at enemies within range, hurling bubbling brew at ships, units and buildings. Build it on the "
                 "brew next to a beach, like a Dock. \n<hp> <attack> <armor> <piercearmor> <range>"}
SHORT_HELP = "Build <b> Cauldron<b> (<cost>) \nBoils over at enemies, hurling bubbling brew at them."

# the cauldron's model, in model units (sculpt.py): an iron pot on a ring of black rocks rising out of the brew,
# three tiles across (half 64), green-blue fire under it
POT_R = 40.0  # the pot's radius
POT_Z = 46.0  # its middle's height
RIM_Z = POT_Z + 22.0  # its rim: where the brew bubbles and the bombs leave it
BREW = (0.49, 1.0, 0.42)
LIME = (0.78, 1.0, 0.54)


# --------------------------------------------------------------------------- the look

def _brew_material(seed=0):
    from . import paint as P
    from . import sculpt as S

    def color(look):
        return P.mix(np.array(LIME), np.array([0.12, 0.55, 0.2]), P.noise(look.p, 2.0, seed))
    return S.Material(color=color, glow=0.9, vivid=1.0)


def volcano(t: Optional[float] = None) -> Part:
    """The cauldron: a great iron pot on black rocks rising out of the brew, bubbling green, fire burning under it,
    green fumes curling up and a team flag on a rock. With `t` (0..1), boiling over: a fountain of brew thrown up,
    highest as the bomb leaves (`FRAME`)."""
    from . import haunted as H
    from . import paint as P
    from . import sculpt as S
    rng = np.random.default_rng(zlib.crc32(b"cauldron"))
    rock = P.mottled(P.hexrgb("#2e2a32"), P.hexrgb("#0e0c10"), 3.0, 0.6, 3)
    iron = P.metal(P.hexrgb("#26242c"), rust=0.4, seed=4, shine=0.7)
    out = []
    for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):  # the rocks it stands on
        r = rng.uniform(40, 56)
        c = np.array([math.cos(a) * r, math.sin(a) * r, 0.0])
        size = rng.uniform(9, 14)
        out.append(H.hull(c + rng.normal(0, size * 0.55, (7, 3)) * np.array([1, 1, 0.9]) + (0, 0, size * 0.6), rock))
    out.append(S.rod((0, 0, 0), (0, 0, POT_Z - 26.0), 34.0, rock, 26.0))
    fire = S.Ellipsoid(S.mat((0.45, 0.95, 0.75), glow=1.0), S._orient((0, 0, POT_Z - 32.0)), radii=(30.0, 30.0, 10.0))
    fire.casts = False
    out.append(fire)
    out.append(S.ball((0, 0, POT_Z), (POT_R, POT_R, POT_R * 0.75), iron).clip((0, 0, 1), RIM_Z - POT_Z))
    out.append(S.torus((0, 0, RIM_Z), POT_R * 0.76, 3.0, iron))
    out.append(S.rod((0, 0, RIM_Z - 1.0), (0, 0, RIM_Z + 1.2), POT_R * 0.72, _brew_material(5)))
    for a in (0.6, 3.7):  # handles: rings of bone
        out.append(S.torus((math.cos(a) * POT_R * 0.95, math.sin(a) * POT_R * 0.95, POT_Z + 6.0), 6.0, 1.4,
                           P.bone(2), rot=(90, 0, math.degrees(a))))
    for k in range(6):  # bubbles
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0, POT_R * 0.6)
        b = S.ball((math.cos(a) * r, math.sin(a) * r, RIM_Z + 1.5), rng.uniform(2.0, 4.5), _brew_material(k))
        b.casts = False
        out.append(b)
    for k in range(3):  # green fumes curling up
        puff = S.Ellipsoid(S.Material(color=lambda l: np.broadcast_to(np.array((0.5, 0.82, 0.58)), (len(l.p), 3)),
                                      glow=0.3, vivid=0.8),
                           S._orient((rng.uniform(-6, 6), rng.uniform(-6, 6), RIM_Z + 14.0 + k * 13.0)),
                           radii=(9.0 + k * 3, 9.0 + k * 3, 6.0 + k * 2))
        puff.casts = False
        puff.ink = False
        out.append(puff)
    out += H.team_flag((44.0, 44.0, 14.0), 24.0, 9)
    root = Part("cauldron", boxes=out)
    if t is not None:
        root.boxes += eruption(t, (0.0, 0.0, RIM_Z))
    return root


def eruption(t: float, at: tuple[float, float, float]) -> list:
    """Brew boiling over at `at`, a moment `t` (0..1) of it: glowing blobs flying up and out and falling back,
    highest as the bomb leaves, a skull or two flung out with them."""
    from . import sculpt as S
    x0, y0, z0 = at
    rng = np.random.default_rng(zlib.crc32(b"cauldron eruption"))
    peak = (FRAME + 0.5) / ERUPTION_FRAMES
    burst = math.exp(-((t - peak) / 0.22) ** 2)
    jet = S.Ellipsoid(_brew_material(1), S._orient((x0, y0, z0 + 6 + 30 * burst)), radii=(10.0, 10.0, 8 + 30 * burst))
    jet.casts = False
    jet.ink = False
    out = [jet]
    for k in range(32):
        phase = (t * 1.5 + k / 32) % 1
        angle = rng.uniform(0, 2 * math.pi)
        reach = rng.uniform(6, 40)
        rise = rng.uniform(60, 150) * (0.35 + 0.65 * burst)
        z = z0 + 10 + 4 * rise * phase * (1 - phase)
        r = reach * phase
        size = rng.uniform(2.5, 5.5) * (1.15 - 0.55 * phase)
        blob = S.ball((x0 + r * math.cos(angle), y0 + r * math.sin(angle), z), size, _brew_material(k))
        blob.casts = False
        blob.ink = False
        out.append(blob)
    return out


def bomb() -> Unit:
    """A blob of boiling brew with a skull in it, drops trailing off it, tumbling as the fireball does (animation rig
    "fireball")."""
    from . import sculpt as S
    from .graveyard import skull
    core = S.ball((0, 0, 0), 5.4, _brew_material(2))
    core.ink = False
    drops_a = [S.ball((4.0, -6.0, 1.0), 1.6, _brew_material(3)), S.ball((-3.0, -8.0, -1.0), 1.2, _brew_material(4))]
    drops_b = [S.ball((-4.0, -6.0, 2.0), 1.6, _brew_material(5)), S.ball((3.0, -8.5, -2.0), 1.2, _brew_material(6))]
    root = Part("root").add(Part("ball", boxes=[core] + skull((0, 2.0, -2.0), 2.4, 3)), Part("flame_a", boxes=drops_a),
                            Part("flame_b", boxes=drops_b))
    return Unit("brew_bomb", "Brew bomb", "the cauldron's shot", "fireball", root, group="easter", attack="none")


def _building_frames(models: list, quant) -> list[slp.SlpFrame]:
    """Pictures of building models from the building view, all through one camera (so they line up), each with the
    footprint's centre on its hotspot."""
    from .export import _crop
    from .sculpt import fit_camera, render
    whole = Part("all", children=models)
    cam = fit_camera(whole, V.BUILDING_HEADING, pad=6)
    return [_crop(quant.frame_codes(render(p, V.BUILDING_HEADING, camera=cam), obstruction=False), cam.origin)
            for p in models]


_SLPS: dict[tuple, bytes] = {}


def volcano_slp(quant) -> bytes:
    """The cauldron at rest: one picture."""
    key = ("rest", np.asarray(quant.palette).tobytes())
    if key not in _SLPS:
        _SLPS[key] = slp.encode(_building_frames([volcano()], quant))
    return _SLPS[key]


def eruption_slp(quant, frames: int = ERUPTION_FRAMES) -> bytes:
    """Boiling over: `frames` pictures of the cauldron throwing up brew, one direction."""
    key = ("erupt", np.asarray(quant.palette).tobytes(), frames)
    if key not in _SLPS:
        _SLPS[key] = slp.encode(_building_frames([volcano(k / frames) for k in range(frames)], quant))
    return _SLPS[key]


def bomb_slp(quant, frames: int = BOMB_FRAMES) -> bytes:
    """The brew bomb: `frames` pictures of it tumbling, one direction, no shadow (it is in the air)."""
    key = ("bomb", np.asarray(quant.palette).tobytes(), frames)
    if key not in _SLPS:
        from .export import render_frames
        _SLPS[key] = slp.encode(render_frames(bomb(), "idle", frames, 1, False, quant, shadow=False))
    return _SLPS[key]


def crater() -> tuple[float, float, float]:
    """Where its bombs leave it (tiles across, forward and up): the cauldron's rim."""
    from .export import BASE_SCALE
    from .gameplay import HEIGHT_PIXELS
    return 0.0, 0.0, round(RIM_Z * math.cos(math.radians(30)) * BASE_SCALE / HEIGHT_PIXELS, 2)


# --------------------------------------------------------------------------- its icon

ICON_COLOURS = {0: (30, 20, 40), 1: (40, 38, 46), 2: (16, 14, 20), 3: (90, 220, 80), 4: (200, 255, 140),
                5: (240, 120, 30), 6: (120, 160, 130), 7: (200, 196, 180)}


def icon(size: int = 36) -> np.ndarray:
    """A building icon (drawn at 36x36, scaled to `size`): the cauldron boiling over against a night sky, green brew
    spilling, fire under it. Colours are keys of ICON_COLOURS."""
    sky, iron, dark, brew, glow, fire, fumes, bone = range(8)
    px = np.full((36, 36), sky, np.int16)
    yy, xx = np.mgrid[0:36, 0:36]
    pot = ((xx - 18) / 14.0) ** 2 + ((yy - 22) / 11.0) ** 2 < 1
    px[pot & (yy >= 15)] = iron
    px[pot & (yy >= 15) & (xx < 12)] = dark
    px[15:17, 4:32] = iron
    px[13:16, 6:30] = brew
    px[12:14, 9:14] = glow
    px[11:13, 20:24] = glow
    px[32:36, 8:28] = fire
    px[33:36, 12:24] = glow
    px[30:36, 2:6] = dark
    px[30:36, 30:34] = dark
    for y, x in ((7, 17), (4, 13), (5, 22), (2, 18), (8, 10), (6, 26)):
        px[y:y + 2, x:x + 2] = brew if (x + y) % 2 else glow
    px[0:4, 3:10] = fumes
    px[0:3, 26:33] = fumes
    px[20:23, 3:6] = bone
    px[20:23, 30:33] = bone
    if size != 36:
        idx = np.arange(size) * 36 // size
        px = px[idx][:, idx]
    return px


def add_icon(data: bytes, index: int, quant) -> tuple[Optional[bytes], str]:
    """A building icon sheet with the cauldron's icon added as icon `index` (blank icons fill any gap before it)."""
    try:
        info = slp.info(data)
    except ValueError as exc:
        return None, f"not changed ({exc})"
    if info.num_frames > index:
        return None, f"not changed: this sheet already has an icon {index}"
    sizes = [(w, h) for w, h, _, _ in info.sizes if 16 <= w <= 96 and 16 <= h <= 96]
    if not sizes:
        return None, "not changed: no icon in this sheet has a usable size"
    w, h = max(set(sizes), key=sizes.count)
    lut = {k: int(quant.indices(np.array([rgb], np.int64))[0]) for k, rgb in ICON_COLOURS.items()}
    px = np.vectorize(lut.get)(icon(min(w, h))).astype(np.int16)
    if (w, h) != px.shape[::-1]:  # not square: centred on the sky
        full = np.full((h, w), lut[0], np.int16)
        y0, x0 = (h - px.shape[0]) // 2, (w - px.shape[1]) // 2
        full[y0:y0 + px.shape[0], x0:x0 + px.shape[1]] = px
        px = full
    blank = slp.SlpFrame(np.full((1, 1), slp.TRANSPARENT, np.int16), (0, 0))
    frames = [blank] * (index - info.num_frames) + [slp.SlpFrame(px, (0, 0))]
    return slp.append_frames(data, frames), f"the cauldron is icon {index} ({w}x{h})"


def icon_sheets(archives: list[tuple[str, object]]) -> list[tuple[str, object, int, bytes]]:
    """Every copy of the building icon sheets, in load order: (archive name, archive, sheet id, sheet)."""
    return [(name, drs, sid, drs.get(sid)) for name, drs in archives if drs is not None
            for sid in BUILDING_ICONS if sid in drs.ids()]


# --------------------------------------------------------------------------- the .dat

def dock(civs):
    """The Dock (as the game calls it: its unit name), whose placement rules the volcano takes."""
    return next((u for units in civs.units for u in units if u is not None and u.type == 80 and u.name == DOCK
                 and u.values["placement_side"][0] >= 0), None)


def _own_graphic(graphics: dict, civs, gid: int, owners: set[int]) -> bool:
    """True if graphic `gid` is drawn by the `owners` units alone, from an SLP of its own, with no per-angle sounds
    (so its angle count may change)."""
    g = graphics.get(gid)
    if g is None or g.slp <= 0 or g.slp_at < 0 or g.angle_sounds_at >= 0:
        return False
    if sum(1 for o in graphics.values() if o.slp == g.slp) != 1 or any(
            d.graphic_id == gid for o in graphics.values() for d in o.deltas):
        return False
    for units in civs.units:
        for u in units:
            if u is None or u.id in owners:
                continue
            looks = {*u.values.get("standing", ()), *u.values.get("dying", ()), *u.values.get("walking", ()),
                     *u.values.get("damage_graphics", ()), u.values.get("attack_graphic", -1)}
            if gid in looks:
                return False
    return True


def _redraw(data: bytearray, g, slp_id: int, frames: int, seconds: float, layer: int, sound: int,
            sequence: int) -> None:
    """Graphic `g` draws SLP `slp_id`, in one direction, played as `sequence`, and nothing else with it: its old
    extra layers (the Sea Tower's shadow, the arrows' shadow on the ground) become the picture itself (delta -1); the
    new pictures have their own shadows."""
    for k, d in enumerate(g.deltas):
        if d.graphic_id != -1:
            struct.pack_into("<h", data, g.delta_at(k), -1)
    struct.pack_into("<i", data, g.slp_at, slp_id)
    struct.pack_into("<b", data, g.layer_at, layer)
    struct.pack_into("<HH", data, g.frames_at, frames, 1)  # one direction
    struct.pack_into("<f", data, g.frame_rate_at, seconds)
    struct.pack_into("<b", data, g.sequence_at, sequence)
    struct.pack_into("<b", data, g.mirroring_at, 0)
    struct.pack_into("<h", data, g.sound_at, sound)


def patch(data: bytearray, civs, graphics: dict, slps: tuple[int, int, int],
          icon_index: Optional[int] = None) -> tuple[Optional[dict], str, Optional[dict[str, int]]]:
    """The Sea Tower becomes the volcano (in place), drawn from SLPs `slps` (at rest, erupting, its bomb), with icon
    `icon_index` if given. Returns (what changed: its row and graphics, or None; what happened; its text ids)."""
    first = next((units[SEA_TOWER] for units in civs.units if len(units) > max(SEA_TOWER, *SHOTS)
                  and units[SEA_TOWER] is not None), None)
    stand = graphics.get(first.values["standing"][0]) if first is not None else None
    if first is None or first.type != 80 or stand is None or not stand.name.upper().startswith(LOOK):
        return None, f"Volcano: not changed, unit {SEA_TOWER} is not the Sea Tower here", None
    harbour = dock(civs)
    if harbour is None:
        return None, "Volcano: not changed, there is no Dock to take the placement rules of", None
    row, side, hill = (harbour.values[k] for k in ("terrain_restriction", "placement_side", "hill_mode"))
    shots = [units[s] for units in civs.units if len(units) > max(SHOTS) for s in SHOTS if units[s] is not None]
    if not shots or any(s.type != 60 for s in shots):
        return None, f"Volcano: not changed, units {SHOTS[0]} and {SHOTS[1]} are not the Sea Tower's shots here", None
    rest = first.values["standing"][0]
    erupt, flight = (next(s for s in shots if s.id == k).values["standing"][0] for k in reversed(SHOTS))
    owners = {SEA_TOWER, *SHOTS}
    if len({rest, erupt, flight}) != 3 or not all(_own_graphic(graphics, civs, g, owners) for g in (rest, erupt, flight)):
        return None, "Volcano: not changed, the Sea Tower's pictures are not its own here", None
    try:
        sounds = datfile.sound_files(bytes(data))
    except (ValueError, struct.error):
        sounds = {}
    boom = next((sid for sid, files in sounds.items() if BOOM in (f.lower() for f in files)), -1)
    by_name = {g.name.upper(): gid for gid, g in graphics.items()}
    site, sinking = by_name.get(SITE), by_name.get(SINKING)
    patched = 0
    for units in civs.units:
        if len(units) <= max(SEA_TOWER, *SHOTS) or units[SEA_TOWER] is None or units[SEA_TOWER].type != 80:
            continue
        u = units[SEA_TOWER]
        armours = [(c, ARMOUR.get(c, a)) for c, a in u.values["armours"]]
        attacks = ATTACK if len(u.values["attacks"]) == len(ATTACK) else None
        DU.patch(data, u, enabled=1, train_location=VILLAGER, button=BUTTON, cost=COST, train_time=TIME,
                 hit_points=HP, terrain_restriction=row, placement_side=side, hill_mode=hill,
                 projectile=SHOTS[0], placement_terrain=(LAVA, LAVA), armours=armours,
                 max_range=RANGE, line_of_sight=SIGHT, reload=RELOAD, attack_graphic=erupt, frame_delay=FRAME,
                 displacement=crater(), collision_size=(SIZE, SIZE, u.values["collision_size"][2]),
                 outline_size=(SIZE, SIZE, u.values["outline_size"][2]), clearance_size=(SIZE, SIZE),
                 displayed=(ARMOUR[4], ATTACK[0][1], RANGE, RELOAD), displayed_pierce=ARMOUR[3])
        if attacks:
            DU.patch(data, u, attacks=attacks)
        if site is not None:
            DU.patch(data, u, construction_graphic=site)
        if sinking is not None:
            DU.patch(data, u, dying=(sinking, -1))
        if icon_index is not None:
            DU.patch(data, u, icon=icon_index)
        for s in (units[k] for k in SHOTS):
            if s is not None and s.type == 60:
                DU.patch(data, s, standing=(flight, -1), walking=(flight, -1), projectile_arc=ARC)
        patched += 1
    _redraw(data, graphics[rest], slps[0], 1, 0.0, BUILDING_LAYER, -1, STILL)
    _redraw(data, graphics[erupt], slps[1], ERUPTION_FRAMES, ERUPTION_SECONDS, BUILDING_LAYER, boom, ERUPTING)
    _redraw(data, graphics[flight], slps[2], BOMB_FRAMES, BOMB_SECONDS, SHOT_LAYER, -1, TUMBLING)
    strings = {"name": first.values["name_id"], "creation": first.values["creation_id"]}
    if first.values["help_id"] > HELP_STRINGS:
        strings["help"] = first.values["help_id"] - HELP_STRINGS
    done = {"row": row, "rest": rest, "erupt": erupt, "flight": flight}
    return done, (f"Volcano: the Sea Tower (unit {SEA_TOWER}) for {patched} civilisations, built by Villagers "
                  f"(button {BUTTON}) on the lava by a beach, on the Dock's rules (unit {harbour.id}: beside terrains "
                  f"{side[0]}, {side[1]}, terrain restriction {row}, hill mode {hill}) with lava (terrain {LAVA}) "
                  f"under its middle: "
                  f"{COST[1]} stone, {COST[4]} gold, {TIME} s, {HP} hit points, 3x3 tiles, lava bombs "
                  f"{'/'.join(str(a) for _, a in ATTACK)} (pierce/ships/buildings) at range {RANGE:g} every "
                  f"{RELOAD:g} s" + (f", icon {icon_index}" if icon_index is not None else "")
                  + f"; at rest SLP {slps[0]} (graphic {rest}), erupting SLP {slps[1]} (graphic {erupt}, "
                    f"{ERUPTION_FRAMES} pictures" + (f", sound {boom}" if boom >= 0 else "") + f"), its bombs "
                    f"(units {SHOTS[0]}, {SHOTS[1]}) SLP {slps[2]} (graphic {flight})"), strings

