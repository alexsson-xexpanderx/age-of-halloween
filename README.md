# Age of Halloween

A mod for **Age of Empires II: Gold Edition** that turns the whole game into a
Halloween night. Phantom villagers glide to work, skeleton soldiers march with
ghoul crossbowmen and imp skirmishers, death knights ride nightmares and the
Paladin is the Headless Horseman. Coffin rams, bone catapults and a gallows
trebuchet lay siege; ghost ships sail under torn sails. Every building is
a haunted house, with crooked roofs, boarded windows, porches, turrets, spider
webs and ravens, in five haunted cultures (Victorian haunted houses,
Transylvanian gothic, haunted desert houses, ghost lanterns and Day of the
Dead) that change with each age, with walls, castles and eighteen haunted
wonders. Farms are pumpkin
patches, forests are twisted and bare, and the menu is a churchyard under a
full moon. All 105 units are modelled. Easter eggs: build a Wonder and it
trains Pac-Man, a scary jack-o'-lantern, and a Bone Dragon breathing green
fire; trade carts are ghost trains laying their own rails, and on the Witch
Brew Islands map villagers build cauldrons that spit brew at enemy ships.

![style](previews/style.png)

The pictures are generated, not hand-drawn. Each unit and building is a model
of smooth shapes (spheres, rods, turned and cut shapes) with painted materials,
drawn by a small ray tracer from AoE2's camera in every direction and
animation frame, with team-colour regions ready for the game's palette.

See [docs/DESIGN.md](docs/DESIGN.md) for the art direction and the path into
the game, and [docs/UNITS.md](docs/UNITS.md) for the full unit list. Age of
Halloween is built on [Age of Minecraft](https://github.com/alexsson-xexpanderx/age-of-minecraft)'s
machinery, and the two install side by side.

## Try it in your game

You need AoE2: The Conquerors with UserPatch 1.5, and Python 3.
Double-click **`build_mod.bat`** (it finds your game or asks for the folder),
then double-click **Age of Halloween** on your desktop. The build reads your own
game files and makes Age of Halloween a game of its own (a UserPatch mod with
its own `age_of_halloween.exe` and a jack-o'-lantern icon), so your normal game
stays untouched. Full steps and options are in [docs/INSTALL.md](docs/INSTALL.md).

## Previews

```sh
pip install -r tools/requirements.txt
python tools/cast_sheet.py           # previews/cast_<group>.png
python tools/town_sheet.py           # previews/town_<group>.png
python tools/wilds_sheet.py          # previews/wilds.png
python tools/style_sheet.py          # previews/style.png
```

| File | Shows |
|---|---|
| `previews/style.png` | the look: Pac-Man and the giant, skeletons, a haunted house, a graveyard wall, dead trees |
| `previews/cast_<group>.png` | every unit of a group (civilians, foot, uniques, cavalry, siege, ships, animals, easter eggs) |
| `previews/town_<group>.png` | the buildings in the five haunted cultures and the ages: houses, halls, military, economy, forts, walls, wonders, building sites |
| `previews/wilds.png` | trees, mines, bushes, rocks, plants and the map decorations |
| `previews/farms.png` | the pumpkin patches, which the game draws as ground: being built, grown and exhausted |
| `previews/menu.png` | the main menu, with the game's texts and buttons drawn over it |
| `previews/loading.png` | the loading screen |
| `previews/screens.png` | the other screens as the crypt, and the loading screen, on stand-in screens |
| `previews/interface.png` | the resource bar and bottom panel, on a stand-in for the game's own panel |

## Layout

```
tools/aoh/sculpt.py      the ray tracer: smooth shapes (spheres, rods, turned shapes, blocks, cut by planes), the
                         night's light, glowing materials, ink lines; draws Age of Minecraft's boxes too
tools/aoh/paint.py       materials: pumpkin, wood, bricks, shingles, bone, bark, team cloth, metal, noise
tools/aoh/geometry.py    model parts and poses (shared with the old box models)
tools/aoh/animation.py   rigs and attack styles: stand / walk / attack / die / carry poses
tools/aoh/creatures.py   bodies (skeleton, zombie, phantom, ghoul, imp, werewolf, mummy, vampire, ghost...) and their gear
tools/aoh/infantry.py    villagers, monk, king, foot soldiers, archers and the foot unique units
tools/aoh/cavalry.py     nightmares and bone horses and their riders, the Headless Horseman
tools/aoh/engines.py     coffin rams, bone catapults and ballistae, the bombard, the trebuchet, the hearse, the ghost train
tools/aoh/ghostships.py  ghost ships, the ferry of the dead, the coffin bomb
tools/aoh/beasts.py      animals, and the ghost, the hot-rod hearse and the Bone Dragon
tools/aoh/pumpkins.py    Pac-Man and the giant Pac-Man as jack-o'-lanterns, and small lanterns
tools/aoh/undead.py      skulls and bones
tools/aoh/roster.py      the full roster and preview groups
tools/aoh/haunted.py     the five haunted cultures (kits per age) and the parts buildings are made of
tools/aoh/spooky.py      the haunted house: storeys, crooked roofs, dormers, porches, turrets, webs, ravens
tools/aoh/estates.py     houses, town center, mill, camps, barracks, range, stable, market, dock, farms, sites
tools/aoh/fortress.py    walls, gates, towers, the outpost and castles
tools/aoh/marvels.py     the eighteen wonders and the scenario monuments
tools/aoh/wilds.py       trees, stumps, mines, bushes, rocks, plants, map decorations and projectiles
tools/aoh/graveyard.py   railings, skulls and dead trees
tools/aoh/farmland.py    farms, which are terrain: pumpkin patches in terrain.drs
tools/aoh/lava.py        the witch's brew (the unused terrain 15), Docks on it, and the Witch Brew Islands map
tools/aoh/rails.py       ghost trains laying their own rails: a track left behind them on the ground
tools/aoh/volcano.py     cauldrons Villagers build in the brew by the beach, spitting brew at enemies (the Sea Tower)
tools/aoh/interface.py   the screen panels repainted: old planks, a bone-pale panel, slots and icons
tools/aoh/nightscape.py  painting the menus: night sky, moon, hills, mist, stone and planks, the title
tools/aoh/trinkets.py    the menu's button objects: a jack-o'-lantern, a skull shield, a spell book, a cauldron...
tools/aoh/menu.py        the main menu: a churchyard at night, every button a Halloween thing, names on boards
tools/aoh/screens.py     the other screens: found by their screen files, redrawn as the crypt
tools/aoh/loadscreen.py  the loading screen: a full moon over a churchyard, in a palette of its own
tools/aoh/props.py       draws buildings and scenery in the original sprite's frame layout
tools/aoh/slp.py         SLP 2.0N sprite encoder/decoder
tools/aoh/drs.py         DRS archive reader/writer
tools/aoh/palette.py     game palette and colour matching
tools/aoh/datfile.py     graphics and terrain table reader for empires2_x1_p1.dat
tools/aoh/datunits.py    reads and patches the civilisations' unit tables in the .dat
tools/aoh/gameplay.py    Pac-Man and the Bone Dragon at the Wonder: the .dat changes, icons, names and sounds
tools/aoh/langdll.py     reads and changes strings in the game's language DLLs
tools/aoh/pe.py          Windows DLL/exe resources: read them, give a copy new ones in an added section
tools/aoh/appicon.py     the jack-o'-lantern icon: the .ico file, and in place in the mod's own exe
tools/aoh/sounds.py      Pac-Man's sounds, synthesised in an 8-bit arcade style
tools/aoh/roars.py       the dragon's sounds: roars, growls, wing beats and fire, synthesised
tools/aoh/slpmap.py      which original sprite id each unit replaces
tools/aoh/spritemap.py   finds buildings, walls, trees and decorations by name in the .dat
tools/aoh/export.py      draws an animation in the original sprite's frame layout
tools/build_mod.py       builds the mod from your game files (build_mod.bat on Windows)
tools/*_sheet.py         the preview sheets; concept_sheet.py draws the menu, screens and panel previews
tools/tests/             format round trips and a full build against a fake game folder
```

The modules Age of Minecraft drew with (units.py, mounted.py, siege.py, ships.py, animals.py, easter.py,
structures.py, voxel.py, blocks.py and the rest) are still in `tools/aoh/`: the Halloween models take their keys
and sizes, and a few tests compare against them.

## Status

All 105 units, every building in the five haunted cultures and the ages, walls
and gates, the wonders, the pumpkin patches, the map's nature and decorations,
the menu, the loading screen, the other screens and the panels are made. The
next step is playing it.
