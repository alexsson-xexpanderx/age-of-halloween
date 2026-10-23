"""The full Age of Halloween roster: every AoE2 Gold Edition unit, by group."""
from __future__ import annotations

from typing import Callable

from .animals import ANIMALS
from .easter import EASTER
from .mounted import MOUNTED
from .ships import SHIPS
from .siege import SIEGE
from .units import FOOT, Unit
from . import cavalry, engines, ghostships, infantry, pumpkins

# the Halloween models made so far, under the keys of the Minecraft ones they replace (the rest are still those)
HALLOWEEN: dict[str, Callable[[], Unit]] = {
    **infantry.FOOT, **cavalry.MOUNTED, **engines.SIEGE, **ghostships.SHIPS,
    "pacman": pumpkins.pacman,  # (the giant is drawn by gameplay.giant_slp, from pumpkins.giant_pacman)
}

ROSTER: dict[str, Callable[[], Unit]] = {**FOOT, **MOUNTED, **SIEGE, **SHIPS, **ANIMALS, **EASTER, **HALLOWEEN}

# Preview sheets: (file stem, title, filter)
SHEETS = [
    ("civilians", "Civilians", lambda u: u.group == "civilian"),
    ("infantry", "Infantry and archers", lambda u: u.group == "foot" and not u.civ),
    ("cavalry", "Cavalry", lambda u: u.group == "cavalry" and not u.civ),
    ("siege", "Siege and wagons", lambda u: u.group == "siege" and not u.civ),
    ("ships", "Ships", lambda u: u.group == "ship" and not u.civ),
    ("uniques", "Unique units", lambda u: u.civ is not None),
    ("animals", "Animals", lambda u: u.group == "animal"),
    ("easter", "Easter eggs (cheat units) and the Dragon", lambda u: u.group == "easter"),
]


def build_all() -> dict[str, Unit]:
    return {key: make() for key, make in ROSTER.items()}
