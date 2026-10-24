# Handoff: Age of Halloween

**To the next Claude session:** read this whole file, then CLAUDE.md (still largely Age of Minecraft's: the game-file
machinery it describes is unchanged; the art is being replaced). Branch `main`.

Repository: https://github.com/alexsson-xexpanderx/age-of-halloween (public, created 2026-10-23)
Zip the player downloads: https://github.com/alexsson-xexpanderx/age-of-halloween/archive/refs/heads/main.zip
Age of Minecraft (the parent mod) lives next to it in `C:\Users\Andrei Alexsson\git\age-of-minecraft`, branch
`claude/busy-ritchie-8lktbr`; its HANDOFF.md has the player's setup and how they like to work (all of it applies).

## What the player asked for (2026-10-23)

"A completely new mod, Halloween themed, called Age of Halloween: all buildings, walls and units, as detailed as
possible, not pixelated like Minecraft." Their choices:

- A **separate repository** (this one), started from Age of Minecraft's code (commit 4d2867f: renamed `aom` -> `aoh`,
  `age_of_minecraft` -> `age_of_halloween`, shortcuts and window title "Age of Halloween"), so both mods install
  side by side.
- Style: first "spooky cartoon", then, after seeing it, **"darker and creepier"**: dim moonlight, a sickly green
  rim, drained colours, grime; glows (carved faces, eyes, windows, candles) stay bright. `sculpt.py` MOOD constants.
- **Pac-Man is a scary pumpkin**, the giant Pac-Man "a bit scary pumpkin" (done: `pumpkins.py`).
- **Five haunted cultures** for the five building sets (W Victorian haunted houses, E Transylvanian gothic,
  M mummy tombs, F ghost lanterns and yokai shrines, X Day of the Dead).
- Also themed: **trees and resources, farms (pumpkin patches), menu and screens, interface panels**.
- Keep all of Age of Minecraft's extras, in Halloween versions: **dragon** (bat-winged), **trains** (a ghost train),
  **volcanoes** (bubbling witch's cauldrons spitting green fire at ships), **Lava Islands** (a sea of glowing green
  witch's brew).
- Public repository.

## How it is drawn

`sculpt.py` is the renderer: exact curved shapes (Ellipsoid, Frustum, Lathe, Convex; `clip` cuts them with
planes), 3x3 samples per pixel, ink lines on depth edges, the dark mood light, glowing materials. It takes the same
`geometry.Part` trees and `Pose`s as the old box renderer, so `animation.py`'s rigs move the new models unchanged;
shapes go in a Part's `boxes` list. `paint.py` has the surfaces (pumpkin ribs, wood, bricks, shingles, bone, bark,
team cloth, metal). export.py, props.py and concept_sheet.py draw with it. The menus are painted with
`nightscape.py` (night sky, moon, hills, mist, stone and planks, the title in a Windows font) and `trinkets.py`
(the menu's button objects as sculpt models).

## The cast

- Villagers: zombie peasants (each job with its tool). Monk: a hooded necromancer. King: a vampire lord.
- Militia line: skeleton soldiers, better armour each age (done: militia, man-at-arms). Spear line: skeleton
  spearmen. Archers: skeleton archers; crossbows: ghouls. Skirmishers: imps with javelins. Hand cannoneer: a ghost
  pirate. Petard: a zombie with a jack-o'-lantern bomb. Eagle warriors: Camazotz bat-men.
- Uniques: Longbowman spectral archer; Woad Raider banshee; Chu Ko Nu jiangshi (hopping vampire); Throwing Axeman
  werewolf; Huskarl draugr; Samurai ghost samurai with an oni mask; War Elephant skeletal elephant; Teutonic Knight
  death knight; Janissary ghoul musketeer; Berserk werewolf berserker; Jaguar and Plumed Archer Day of the Dead
  calavera warriors.
- Cavalry: zombie riders on skeletal horses (scout line), death knights on nightmares (knight line; the Paladin is
  the Headless Horseman with a jack-o'-lantern), ghost horse archers, mummy camel riders; uniques to match.
- Siege: coffin rams, bone catapults flinging flaming skulls, bone ballistae, a gargoyle-mouth bombard, a gallows
  trebuchet with a pumpkin payload, a hearse war wagon; the trade cart is the ghost train.
- Ships: ghost ships with tattered sails in the owner's colour and green lanterns; the transport is a ferry of the
  dead; demolition ships floating coffins.
- Animals: nightmare horses, wolves with glowing eyes, black sheep, skeletal stags, demon boars, ravens and bats,
  bony fish.
- Easter: the VDML ghost a bedsheet ghost in the owner's colour; the Cobra car a hot-rod hearse; the dragon a
  bat-winged bone dragon.

## Progress

- 2026-10-23: repo, renderer, materials, the look test (Pac-Man and the giant, two skeletons, a Victorian house, a
  graveyard wall, dead trees), darker mood.
- 2026-10-23/24: all 105 units; every building in the five cultures and ages, walls, gates, towers, castles, the
  wonders; trees, mines and decorations; the cauldron (volcano), the witch's brew (lava) and its map, pumpkin patch
  farms, the texts (Ghost Train, Bone Dragon, Cauldron); the menu (a churchyard at night), the loading screen (a
  full moon), the other screens (the crypt), the panels (planks, a bone-pale panel, sculpt icons), the app icon (the
  giant's jack-o'-lantern) and the Bone Dragon's unit icon; docs.

## Next

- Build into the player's game (`build_mod.bat`; the first build opens SetupAoC.exe for "Install") and look at it
  in game: the menu's button places, the screens, the panels' readability, unit sizes.
- Not yet drawn as Halloween: the Pac-Man and giant unit icons (Age of Minecraft's: hands-off without asking).
