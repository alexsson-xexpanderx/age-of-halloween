# Trying Age of Halloween in your game

The mod is built **on your own PC, from your own game files**. The build
reads your game's colour palette, how many frames each original sprite has,
and the graphics table in your `.dat` file. It then draws every Halloween
sprite to match exactly. Your original files are not changed.

You need **Age of Empires II: The Conquerors** (Gold Edition), **UserPatch
1.5**, and **Python 3**.

## 1. Install Python (one time)

1. Download Python 3 from <https://www.python.org/downloads/> and run the
   installer. Tick **"Add python.exe to PATH"**.
2. That's all. The helper script installs the one library it needs
   (`numpy`) by itself.

## 2. Get the mod

Download this repository as a ZIP (on GitHub: **Code → Download ZIP**) and
extract it anywhere, for example to your Desktop.

## 3. Build it

Open the extracted folder and **double-click `build_mod.bat`**. Leave it in
that folder: it needs the `tools` folder next to it.

* It looks for your game in the usual places, for example
  `C:\Program Files (x86)\Microsoft Games\Age of Empires II`.
* If it can't find the game, it asks for the folder. Open your
  **Age of Empires II** folder in File Explorer (the one with the `Data`
  folder inside), click the address bar at the top, copy the path, paste it
  into the black window and press Enter.
* Tip: if you extract the mod *inside* your Age of Empires II folder, it
  always finds the game.

The build takes about two to ten minutes, depending on your PC. It makes
**Age of Halloween** a game of its own: your normal Age of Empires exe and
files are not changed. It writes:

```
Games\age_of_halloween.xml                    the UserPatch mod definition ("Age of Halloween")
Games\age_of_halloween\Data\gamedata_x1_p1.drs  every Halloween sprite, farm, panel, the menu and screens, the witch's brew, the trains' rails, the cauldron, Pac-Man's and the Bone Dragon's icons and sounds
Games\age_of_halloween\Data\empires2_x1_p1.dat  its rules: Pac-Man and the Bone Dragon at the Wonder, the brew, ghost trains laying rails, cauldrons, the Javelina's own look
Games\age_of_halloween\Script.RM\Witch Brew Islands.rms  the Witch Brew Islands map
Games\age_of_halloween\Data\language_x1_p1.dll  its own texts: the names of Pac-Man, the Bone Dragon, the Ghost Train and the Cauldron, and "Age of Halloween"
Games\age_of_halloween\age_of_halloween.ico    its icon, a jack-o'-lantern
Games\age_of_halloween\menu_originals\         your game's main menu pictures (for the Halloween menu)
Games\age_of_halloween\screen_originals\       your game's other screens (setup, loading, dialogues...)
Games\age_of_halloween\aoh_report.txt         what was replaced, what was skipped, and why
age2_x1\age_of_halloween.exe                  its own exe, made by UserPatch's SetupAoC.exe: its window says "Age of Halloween"
Age of Halloween (shortcut)                   on your desktop and in the game folder, with the icon
```

> If your game is under *Program Files*, Windows may block writing there and
> the window says so. Then right-click `build_mod.bat` and choose
> **Run as administrator**.

## 4. Play

Double-click **Age of Halloween** on your desktop (or in your game folder).
It starts `age2_x1\age_of_halloween.exe`. Your normal game starts as always
and stays exactly as it was.

**Used `build_mod_direct.bat` before?** That one changed your normal game's
files. Double-click **`restore_original.bat`** once to give your normal game
its original files back; Age of Halloween doesn't need them.

**The UserPatch window.** The first build opens it to make
`age_of_halloween.exe`: click its **Install** button and wait; the build waits
for it too. Later builds keep that exe and don't open the window again.

**No `age_of_halloween.exe`?** The exe is made by UserPatch's installer,
`SetupAoC.exe`, which must be in your game folder (it comes with UserPatch 1.5
from <https://userpatch.aiscripters.net/>). Without it the shortcut still
works: it starts your normal exe with `GAME=age_of_halloween`, which tells
UserPatch to load Age of Halloween instead. The end of the black window (under
**RESULT**) says which one you got; everything shown there is also saved to
**`aoh_build_log.txt`**, next to `build_mod.bat`.

**Rather have the Halloween look in your normal game?** Double-click
**`build_mod_direct.bat`**: it puts everything into your normal game's files,
each one backed up first, and **`restore_original.bat`** puts them back.

## Options

| Command | What it does |
|---|---|
| `build_mod.bat "<game>" --only militia,archer,villager` | build only some units, for a quick test (keys are listed in [UNITS.md](UNITS.md)) |
| `build_mod.bat "<game>" --only buildings,farms,walls,wonders,nature,decorations,projectiles` | build only buildings and scenery (any of these groups; `buildings` includes the farms) |
| `build_mod.bat "<game>" --only interface` | build only the Halloween panels at the top and bottom of the screen |
| `build_mod.bat "<game>" --dry-run` | only plan and write `aoh_report.txt` |
| `test_1_rails_only.bat` | a test build with only the trains' rails (the Trade Cart keeps its horse cart look): double-click, then `build_mod.bat` for the whole mod again |
| `test_2_train_look_only.bat` | a test build with only the train's look (no rails): double-click, then `build_mod.bat` for the whole mod again |
| `build_mod_direct.bat` | put the sprites straight into `Data\graphics.drs`, the farms into `Data\terrain.drs` and the panels into `Data\interfac.drs` (no mod exe needed); each file is backed up first as `<name>.aoh-backup` |
| `restore_original.bat` | undo `build_mod_direct.bat` (graphics, farms, panels, the rules file, the icons, the language files and the Witch Brew Islands map) |
| `build_mod.bat "<game>" --no-wonder-pacman` | keep the game's rules unchanged (no Pac-Man or Bone Dragon at the Wonder, no brew, no rails behind the trains, no cauldrons, and the cheat unit keeps its name and icon) |
| `build_mod.bat "<game>" --no-dat` | do not touch `empires2_x1_p1.dat` at all (no Pac-Man or Bone Dragon at the Wonder, no brew, no rails, no cauldrons, and the Javelina keeps the Wild Boar's look) |

## What to expect in this test

* All unit sprites are replaced: standing, walking, attacking, dying and
  decaying, plus villager work and carry animations: phantom villagers,
  skeleton armies, death knights on nightmares, ghost ships and more. Male and
  female villagers share the phantom models.
* All buildings are replaced by haunted houses: crooked steep roofs, boarded
  windows and broken shutters, porches with jack-o'-lanterns, turrets,
  chimneys, spider webs, ravens, broken fences and graves. They come in five
  haunted cultures (Victorian haunted houses, Transylvanian gothic, haunted
  desert houses, ghost lanterns, Day of the Dead) and every age, with walls,
  gates, towers, castles, docks and the eighteen wonders. No pyramids.
* Farms are pumpkin patches. A farm is part of the ground in this game, so
  the build replaces its textures in `terrain.drs`. While a villager builds a
  farm, vines sprout and small green pumpkins grow. A finished farm has big
  ripe pumpkins, and an exhausted farm is cracked earth with rotten pumpkins.
  The edges blend into the grass as the original farms do.
* The main menu is a churchyard at night under a full moon: the buttons are a
  hanging jack-o'-lantern, a spell book on a lectern, a skull shield on
  crossed scythes, an old map, a witch's cauldron, a crystal ball and more, in
  the same places as before, each name on a board of dark planks, under an
  "AGE OF HALLOWEEN" title. The Single Player menu opens on a dark stone wall
  with cobwebs, with dark purple buttons.
* The loading screen is a huge full moon with a haunted house, a churchyard
  and glowing pumpkins; the other screens (game setup, history, achievements,
  dialogues) are dark stone or old planks around purple-grey windows.
* The panels at the top and bottom of the screen are old planks around a
  bone-pale panel, the dark area behind the minimap a dark slot. The resource
  icons are a log (wood), a pumpkin (food), gold coins, a stone and a skull
  (population), all the same size and each as far from its amount. If a
  panel's icons can't be found, that panel keeps its original look, and the
  report says so.
* The map is replaced too: twisted and bare trees, autumn woods, chopped
  trees, gold and stone mines, bramble bushes, rocks, plants, animals, fish
  and decorations (witch's huts, ruined chapels, skulls on pikes, green
  torches and more).
* Projectiles: bone-tipped arrows and bolts, spinning axes, flaming skulls and
  pumpkins. Their ground shadows are the game's own.
* Original sprites are built in layers (shadows, flags, roof pieces, sails).
  The build reads these layers from your `.dat` and hides the extra ones, so
  only the Halloween sprite shows.
* The Elite Eagle Warrior uses the Eagle Warrior's sprites, as in the
  original game.
* The Javelina borrows the Wild Boar's sprites in the original game (its own
  were never shipped). The mod renders its own pig sprites and points the
  Javelina at them in `empires2_x1_p1.dat`, with a pig carcass that holds the
  same food as the boar's.
* A few original sprite files hold a frame or two more than the game uses.
  The build follows the game's own frame layout for those and fills the
  unused frames, so every direction still lines up.
* Still original: cliffs, bridges, the rest of the terrain, fire and explosions, ships sinking,
  and the campaign maps. Unit names are unchanged, except Pac-Man's, the giant's, the Bone Dragon's, the Ghost
  Train's and the Cauldron's.

## Easter eggs

**Pac-Man at the Wonder.** Once you have built a Wonder, select it: its
first button trains **Pac-Man** (500 food, 500 gold, 5 minutes, 250 hit
points; he takes 1 population like any unit), a scary carved jack-o'-lantern
whose top half opens like a jaw full of teeth. He is big, taller than a
knight on horseback, and takes as much room as a Mangonel, so nothing stands
inside him, and he still fits through gates. He has his own Pac-Man icon, and
the game calls him **Pac-Man**. He is faster than any unit in the game, even
the Cobra Car cheat: 5 tiles a second, more than three times a Hussar. This
works for every civilisation. He is infantry, not a wild animal like the cheat
unit: he boards Transport Ships (and gets off again on any shore), walks on
beaches, garrisons like a foot soldier, and gets the
Blacksmith's infantry upgrades and your civilisation's infantry bonuses.

He has his own arcade sounds: click on him and he goes "waka-waka" (or
boings, or giggles), orders get a "wakawakawaka" or a big CHOMP and
"nom nom", the Wonder plays a little jingle when he is ready, every bite
chomps, and he dies with a sad "wah wah wah waaah" and two pops. Listen to
them in Age of Minecraft's `previews/sounds/`. He, the Bone Dragon, the brew, the cauldrons and the trains below are
the mod's changes to the game's rules; `--no-wonder-pacman` leaves the rules
alone.

**The Bone Dragon at the Wonder.** The Wonder's second button trains a **Bone Dragon**
(300 food, 300 gold, 2 minutes, 600 hit points, 1 population), for every
civilisation: a dragon's skeleton with green fire in its ribs and eyes and
tattered wings in your team colour, always up in the air, its shadow on the
ground below. It flies over land, water and the deep sea, so it needs no
ship. It breathes balls of green fire with a burning skull in them: 40 damage every 3 seconds, at a range of 6 tiles,
and its fire burns buildings as well as units. Armour: 4 against melee, 6
against arrows. It flies at 1.4 tiles a second, as fast as Light Cavalry. It
counts as an archer, so the Blacksmith's archer upgrades give it more range.
It has its own sounds: it growls, snorts or rumbles when you click on it,
beats its wings when it moves, roars when you send it to attack and when the
Wonder trains it, every fireball goes "fwoosh", and it dies with a falling
roar, a thud as it hits the ground and a last groan. Its icon is its bony
head breathing green fire. It lives in a slot the
game has but never uses (the "Advanced Heavy Crossbowman"), so no other unit
changes. With `build_mod_direct.bat` most of its texts are too long for the
game's own language files, so only its name changes there.

**Witch Brew Islands.** A new map, like Team Islands but with a sea of
glowing green witch's brew instead of water (swirls, bubbles and froth): each
team shares an island, and the brew around it shows green on the minimap. In
the game setup, set the map style to **Custom** and pick **Witch Brew
Islands**. Ships sail on the brew as they would on water: build your Docks on
the brew by your island's beach (every island has one, all round), then
Transport Ships, warships and Fish Traps. Nothing lives in the brew, so there
are no fish; each island has more berries,
deer, boar and sheep instead. Like Team Islands, the islands are grassland,
desert, mountains or snow, at random. One thing changes on every map: Docks
can no longer be built in shallow fords (they can on the brew instead). With
`build_mod_direct.bat` the map goes into your game's `Random` folder, and
`restore_original.bat` removes it again.

**Cauldrons in the brew.** Select a Villager and open **Build Military
Building**: **Build Cauldron** is in the top row, after the Siege Workshop (150
stone, 100 gold). Place it like a Dock: in the brew, touching your island's
beach (it can't stand on land or water); it is three tiles across. When
enemies come within 8 tiles it boils over every 3 seconds and lobs a blob of
green brew at them: it hurts ships most, but also soldiers and villagers on
the shore and buildings in range. The tower upgrades (Fletching, Bodkin Arrow,
Bracer, Masonry, Architecture, Heated Shot) make it stronger too. The
computer players don't build cauldrons.

**Ghost trains.** The Trade Cart is now a **Ghost Train**: a big black steam
engine with a cowcatcher of bones, its firebox glowing green. It trades as the Trade
Cart did: build it at your Market, then right-click another player's Market.
It lays its own rails as it drives: they appear behind it all the way to the
other Market, round buildings and trees as it goes round them. Like wheel
tracks in the snow, they fade 30 seconds after a train passed (rails that
lasted longer crashed the game when a train was blocked). Computer players'
trains trade and lay rails too.

These names live in the game's language files (`language_x1_p1.dll`,
`language_x1.dll`, `language.dll`, in the game's main folder). Age of
Halloween has its own `language_x1_p1.dll` with them added, so your
game's files stay as they are. `build_mod_direct.bat` renames them in the
game's own files instead, keeping a backup of each file it changes, and
`restore_original.bat` puts them back. Windows 11's Smart App Control refuses
language files that were changed ("Bad Image", error 0xc0e90002); if you see
that, Smart App Control has to be off.

> Multiplayer: because the rules change, every player needs the same build
> of the mod, or the game goes out of sync.

**Cheats.** Four hidden cheat units are replaced. In a single-player game,
press **Enter**, type the cheat and press **Enter** again; the unit appears
next to your Town Center:

| Cheat | You get |
|---|---|
| `furious the monkey boy` | **Pac-Man**, the carved jack-o'-lantern with a brain in his mouth, who chomps as he runs and attacks |
| `i love the monkey head` | a bedsheet **ghost** in your team colour |
| `how do you turn this on` | the Cobra Car as a **hot-rod hearse** in your team colour, flames painted on its sides |
| `to smithereens` | a **giant Pac-Man**, a red haunted castle that sits on its rock and, when it walks, rises and strides on six spider legs: Pac-Man's bite and armour, 10000 hit points, and he no longer blows himself up. He boards and leaves Transport Ships, and takes as much room on the map as a Mangonel, so he fits through gates; his picture covers everything near him |

In a multiplayer game, cheats only work if "Allow cheats" is ticked in the
game setup.

**Please send back** `Games\age_of_halloween\aoh_report.txt`, the pictures in
`Games\age_of_halloween\menu_originals\` and `screen_originals\` (zip the
folders) and a few screenshots. The report lists your game's full graphics table, so anything
that looks off can be mapped exactly in the next round.
