"""Age of Halloween's people on foot: soldiers, villagers, the necromancer monk and the vampire king, the unique units,
and the skeletal war elephant. Built from creatures.py; the keys are Age of Minecraft's (and the game's sprite
targets in slpmap.py).

    Militia line        skeletons, better armour each age; the Champion's black blade glows with runes
    Spearmen            skeletons with spear, pike and halberd
    Eagle Warriors      Camazotz, the bat god's warriors, with obsidian clubs
    Archers             skeleton archers; crossbows are ghouls; Skirmishers are imps with javelins
    Hand Cannoneer      a ghost pirate with a blunderbuss
    Petard              a zombie lifting a jack-o'-lantern bomb with a lit fuse
    Villagers           phantoms: pale glowing spectres in hooded shrouds, with their tools and loads
    Monk                a necromancer: a sculpted skull with red eyes under a black cowl, a staff with a green
                        flame
    King                a vampire lord: slicked-back hair, pointed ears, fangs, a crown, a high collar, a cape
    Uniques             Longbowman: a spectral archer. Woad Raider: a banshee. Chu Ko Nu: a jiangshi.
                        Throwing Axeman: a werewolf. Huskarl: a draugr. Samurai: a ghost samurai with an oni mask.
                        War Elephant: an elephant skeleton with a howdah. Teutonic Knight: a death knight.
                        Janissary: a mummy musketeer. Berserk: a werewolf berserker. Jaguar Warrior and Plumed
                        Archer: Day of the Dead calaveras.
"""
from __future__ import annotations

import math

import numpy as np

from . import creatures as C
from . import paint as P
from . import pumpkins as PK
from . import sculpt as S
from .creatures import Gear, body, person
from .geometry import Part
from .units import Unit


def _unit(key, name, replaces, kind, gear, attack="chop", civ=None, seed=0, group="foot", rig="biped", **bkw):
    def make():
        return person(key, name, replaces, body(kind, seed, **bkw), gear, attack=attack, group=group, civ=civ,
                      seed=seed, rig=rig)
    return make


def petard() -> Unit:
    """A zombie lifting a jack-o'-lantern bomb, its fuse lit."""
    u = person("petard", "Pumpkin Bomber", "Petard", body("zombie", 17), Gear(cloth="rags"), attack="bomb", seed=17)
    arm_r = u.root.find("arm_r")
    b = body("zombie")
    centre = np.array([0.0, 3.4, b.shoulder - 7.6])
    bomb = PK.lantern(centre - (0, 0, 2.8), 3.2, seed=5)
    fuse = S.rod(centre + (0, 0, 2.2), centre + (0.6, -0.4, 4.0), 0.18, S.mat(P.hexrgb("#2a2420")))
    spark = S.ball(centre + (0.6, -0.4, 4.2), 0.55, S.mat(P.hexrgb("#fff0a0"), glow=1.0))
    spark.casts = False
    arm_r.add(Part("item_r", pivot=tuple(centre), boxes=[*bomb, fuse, spark]))
    return u


def war_elephant() -> Unit:
    """An elephant's skeleton with a team-coloured howdah on its back and a skeleton archer riding in it."""
    bone = P.bone(23)
    from .undead import _ribs_material, _skull_material
    big = 1.0
    hip = 22.0
    root = Part("root")
    torso = Part("torso", pivot=(0, 0, hip))
    # spine and ribcage along +y
    ribs = S.ball((0, 1.0, hip + 3.0), (8.0, 13.0, 9.0), _ribs_material(23)).clip((0, 0, -1), 2.0)
    spine = [S.ball((0, y, hip + 11.0 - abs(y) * 0.12), (1.4, 1.2, 1.2), bone) for y in np.linspace(-12, 12, 9)]
    pelvis = S.ball((0, -11.0, hip + 2.0), (6.0, 3.0, 4.5), bone)
    torso.boxes += [ribs, *spine, pelvis]
    # the howdah: a team-coloured canopy on poles over a wooden box
    wood = P.wood(P.hexrgb("#4a3020"), P.hexrgb("#22140c"), along=1)
    torso.boxes += [S.block((-6.5, -6.0, hip + 11.5), (6.5, 6.0, hip + 15.5), wood)]
    for x in (-6, 6):
        for y in (-5.5, 5.5):
            torso.boxes.append(S.rod((x, y, hip + 15.5), (x, y, hip + 25.0), 0.4, wood))
    torso.boxes.append(S.turned((0, 0, hip + 25.0), [(0, 6.0), (9.4, 0.0), (9.4, -0.6), (0, -0.6)],
                                C.team_cloth(0.55, 23)))
    rider = person("howdah_rider", "", "", body("skeleton", 31), Gear(helmet="hood", left="bow"), seed=31).root
    rider_part = Part("rider", offset=(0, 0, hip + 11.0)).add(*rider.children)

    def rename(part):  # its own part names, or the elephant's head turns would turn the rider's head too
        part.name = "rider_" + part.name
        for c in part.children:
            rename(c)
    for c in rider_part.children:
        rename(c)
        if c.name.startswith("rider_leg"):
            c.boxes = []
    torso.add(rider_part)
    # the head: a big skull with tusks and a trunk of bone rings
    head = Part("head", pivot=(0, 12.0, hip + 6.0))
    skull_m = C.eyes_material(bone, [(-3.0, 1.5, 1.2, 1.0), (3.0, 1.5, 1.2, 1.0)], (1.0, 0.35, 0.15), 0.5)
    head.boxes += [S.Ellipsoid(skull_m, S._orient((0, 17.0, hip + 7.0)), radii=(6.0, 6.0, 6.5))]
    for sx in (-1, 1):
        pts = [np.array([sx * 3.2, 20.0, hip + 2.0]), np.array([sx * 4.0, 25.0, hip - 1.0]),
               np.array([sx * 3.4, 28.0, hip + 2.5])]
        for a, c in zip(pts, pts[1:]):
            head.boxes += S.limb(a, c, 1.2, S.mat(P.hexrgb("#e8dcb8"), shine=0.5), 0.6)
    for k in range(7):
        a = k / 6
        c = np.array([0, 22.0 + a * 3.0, hip + 3.0 - a * 16.0 + a * a * 4.0])
        head.boxes.append(S.torus(tuple(c), 1.6 - a * 0.7, 0.55, bone, rot=(70 - a * 40, 0, 0)))
    torso.add(head)

    def leg(name, x, y):
        knee = (x, y + 0.6, hip * 0.5)
        return Part(name, pivot=(x, y, hip), boxes=[
            *C._limb_shapes(body("skeleton", 23), (x, y, hip), knee, 2.6, 2.2),
            *C._limb_shapes(body("skeleton", 23), knee, (x, y, 2.0), 2.2, 2.0),
            S.rod((x, y, 0.0), (x, y, 2.4), 3.0, bone, 2.6)])

    root.add(torso, leg("leg_fr", 5.0, 9.0), leg("leg_fl", -5.0, 9.0), leg("leg_br", 5.0, -9.0),
             leg("leg_bl", -5.0, -9.0))
    tail = Part("tail", pivot=(0, -13.0, hip + 4.0), boxes=[
        S.ball((0, -13.0 - k * 1.4, hip + 4.0 - k * 1.6), 0.7 - k * 0.07, bone) for k in range(6)])
    root.add(tail)
    return Unit("war_elephant", "Elephant Skeleton", "War Elephant", "quadruped", root, group="foot",
                attack="bite", civ="Persians", scale=0.8)


FOOT = {
    # villagers: phantoms, pale glowing spectres in hooded shrouds with red eyes (the zombies looked robotic)
    "villager": _unit("villager", "Phantom Villager", "Villager", "phantom",
                   Gear(cloth="shroud"),
                   rig="villager", group="civilian", seed=1),
    "villager_lumberjack": _unit("villager_lumberjack", "Phantom Lumberjack", "Lumberjack", "phantom",
                   Gear(cloth="shroud", right="woodaxe", carry="wood"),
                   group="civilian", seed=2),
    "villager_gold_miner": _unit("villager_gold_miner", "Phantom Gold Miner", "Gold Miner", "phantom",
                   Gear(cloth="shroud", right="pickaxe", carry="gold"),
                   group="civilian", seed=3),
    "villager_stone_miner": _unit("villager_stone_miner", "Phantom Stone Miner", "Stone Miner", "phantom",
                   Gear(cloth="shroud", right="pickaxe", carry="stone"),
                   group="civilian", seed=4),
    "villager_farmer": _unit("villager_farmer", "Phantom Farmer", "Farmer", "phantom",
                   Gear(cloth="shroud", right="hoe", carry="food"),
                   group="civilian", seed=5),
    "villager_builder": _unit("villager_builder", "Phantom Builder", "Builder", "phantom",
                   Gear(cloth="shroud", right="hammer"),
                   group="civilian", seed=6),
    "villager_repairer": _unit("villager_repairer", "Phantom Repairer", "Repairer", "phantom",
                   Gear(cloth="shroud", right="hammer"),
                   group="civilian", seed=7),
    "villager_hunter": _unit("villager_hunter", "Phantom Hunter", "Hunter", "phantom",
                   Gear(cloth="shroud", left="bow", back="quiver", carry="meat"),
                   attack="bow", group="civilian", seed=8),
    "villager_fisherman": _unit("villager_fisherman", "Phantom Fisherman", "Fisherman", "phantom",
                   Gear(cloth="shroud", right="rod", carry="fish"),
                   attack="fish", group="civilian", seed=9),
    "villager_forager": _unit("villager_forager", "Phantom Forager", "Forager", "phantom",
                   Gear(cloth="shroud", left="basket", carry="berries"),
                   attack="gather", group="civilian", seed=10),
    "villager_shepherd": _unit("villager_shepherd", "Phantom Shepherd", "Shepherd", "phantom",
                   Gear(cloth="shroud", right="lantern", carry="meat"),
                   attack="shear", group="civilian", seed=11),
    "monk": _unit("monk", "Necromancer", "Monk", "skeleton", Gear(helmet="cowl", cloth="robe", right="staff",
                                                                    left="orb"), attack="cast", group="civilian",
                  seed=12, sculpted=True, eye=C.RED_EYES),
    "king": _unit("king", "Vampire Lord", "King", "vampire", Gear(helmet="crown", cloth="robe", cape=True),
                  group="civilian", seed=13),
    # the militia line: skeletons
    "militia": _unit("militia", "Skeleton Militia", "Militia", "skeleton",
                     Gear(cloth="loin", right="sword", shield="round"), seed=0),
    "man_at_arms": _unit("man_at_arms", "Skeleton Man-at-Arms", "Man-at-Arms", "skeleton",
                         Gear(helmet="nasal", cloth="tabard", armour="plate", right="sword", shield="round"), seed=1),
    "long_swordsman": _unit("long_swordsman", "Skeleton Long Swordsman", "Long Swordsman", "skeleton",
                            Gear(helmet="kettle", cloth="tabard", armour="mail", right="longsword", shield="kite"),
                            seed=2),
    "two_handed": _unit("two_handed", "Skeleton Two-Handed Swordsman", "Two-Handed Swordsman", "skeleton",
                        Gear(helmet="great", cloth="tabard", armour="plate", right="greatsword"), seed=3),
    "champion": _unit("champion", "Skeleton Champion", "Champion", "skeleton",
                      Gear(helmet="horned", armour="plate", cape=True, right="runesword"), seed=4),
    "spearman": _unit("spearman", "Skeleton Spearman", "Spearman", "skeleton",
                      Gear(helmet="hood", cloth="tunic", right="spear"), attack="thrust", seed=5),
    "pikeman": _unit("pikeman", "Skeleton Pikeman", "Pikeman", "skeleton",
                     Gear(helmet="kettle", cloth="tabard", armour="mail", right="pike"), attack="thrust", seed=6),
    "halberdier": _unit("halberdier", "Skeleton Halberdier", "Halberdier", "skeleton",
                        Gear(helmet="great", cloth="tabard", armour="plate", right="halberd"), attack="thrust", seed=7),
    "eagle_warrior": _unit("eagle_warrior", "Camazotz Warrior", "Eagle Warrior", "batman",
                           Gear(cloth="loin", right="club"), seed=8),
    "elite_eagle_warrior": _unit("elite_eagle_warrior", "Elite Camazotz Warrior", "Elite Eagle Warrior", "batman",
                                 Gear(cloth="tunic", helmet="feathers", right="club", shield="round"), seed=9),
    "archer": _unit("archer", "Skeleton Archer", "Archer", "skeleton",
                    Gear(helmet="hood", cloth="tunic", left="bonebow", back="quiver"), attack="bow", seed=10),
    "crossbowman": _unit("crossbowman", "Ghoul Crossbowman", "Crossbowman", "ghoul",
                         Gear(cloth="rags", chest="crossbow"), attack="crossbow", seed=11),
    "arbalest": _unit("arbalest", "Ghoul Arbalest", "Arbalest", "ghoul",
                      Gear(helmet="kettle", cloth="tabard", armour="mail", chest="crossbow", cape=True),
                      attack="crossbow", seed=12),
    "skirmisher": _unit("skirmisher", "Imp Skirmisher", "Skirmisher", "imp", Gear(cloth="loin", right="javelin"),
                        attack="throw", seed=13, size=1.2, hunch=8),
    "elite_skirmisher": _unit("elite_skirmisher", "Elite Imp Skirmisher", "Elite Skirmisher", "imp",
                              Gear(cloth="tunic", right="javelin", shield="round"), attack="throw", seed=14,
                              size=1.2, hunch=8),
    "hand_cannoneer": _unit("hand_cannoneer", "Ghost Pirate", "Hand Cannoneer", "ghost",
                            Gear(helmet="tricorn", cloth="tunic", chest="blunderbuss"), attack="gun", seed=15),
    "petard": petard,
    # unique units
    "longbowman": _unit("longbowman", "Spectral Longbowman", "Longbowman", "ghost",
                        Gear(helmet="hood", cloth="tunic", left="longbow", back="quiver"), attack="bow",
                        civ="Britons", seed=18),
    "woad_raider": _unit("woad_raider", "Banshee", "Woad Raider", "banshee", Gear(cloth="dress"), attack="punch",
                         civ="Celts", seed=19),
    "chu_ko_nu": _unit("chu_ko_nu", "Jiangshi", "Chu Ko Nu", "jiangshi", Gear(cloth="wraps", chest="repeater"),
                       attack="crossbow", civ="Chinese", seed=20, arms_out=75),
    "throwing_axeman": _unit("throwing_axeman", "Werewolf Axeman", "Throwing Axeman", "werewolf",
                             Gear(cloth="loin", right="axe"), attack="throw", civ="Franks", seed=21),
    "huskarl": _unit("huskarl", "Draugr", "Huskarl", "draugr",
                     Gear(helmet="viking", armour="mail", cloth="tunic", right="bigaxe", shield="round"),
                     civ="Goths", seed=22),
    "samurai": _unit("samurai", "Ghost Samurai", "Samurai", "ghost",
                     Gear(helmet="oni", armour="lamellar", cloth="hakama", right="katana"), civ="Japanese",
                     seed=24),
    "war_elephant": war_elephant,
    "teutonic_knight": _unit("teutonic_knight", "Death Knight", "Teutonic Knight", "knight",
                             Gear(armour="plate", cape=True, right="greatsword"), civ="Teutons", seed=25),
    "janissary": _unit("janissary", "Mummy Janissary", "Janissary", "mummy", Gear(helmet="fez", cloth="tunic",
                                                                                     chest="gun"),
                       attack="gun", civ="Turks", seed=26),
    "berserk": _unit("berserk", "Werewolf Berserker", "Berserk", "werewolf",
                     Gear(cloth="loin", right="bigaxe", shield="round"), civ="Vikings", seed=27),
    "jaguar_warrior": _unit("jaguar_warrior", "Calavera Jaguar Warrior", "Jaguar Warrior", "calavera",
                            Gear(helmet="pelt", cloth="loin", right="club", shield="round"), civ="Aztecs", seed=28),
    "plumed_archer": _unit("plumed_archer", "Calavera Plumed Archer", "Plumed Archer", "calavera",
                           Gear(helmet="feathers", cloth="loin", left="bow", back="quiver"), attack="bow",
                           civ="Mayans", seed=29),
}
