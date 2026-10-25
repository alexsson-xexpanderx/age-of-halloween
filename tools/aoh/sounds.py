"""Pac-Man's sounds: the voice of a carved jack-o'-lantern, deep, hollow and wicked, synthesised from scratch (nothing
is sampled). The player asked for scary sounds, definitely not Pac-Man's (the 8-bit "waka-waka" of Age of
Minecraft): there is no chomp tune, jingle or "wah wah" left.

His voice is built like the Dragon's (roars.py): a throat's buzz, unsteady and a little rasping, through a mouth's
resonances. But the mouth says vowels, "ha", "heh", "ooo", lowered as from a big head, with a hollow boom under them
as if he spoke from inside a pumpkin; then it echoes. Laughter is short syllables, each a breath "h" and a vowel,
falling in pitch. A whisper is breath alone through the vowels. A creak is slow clicks ringing in wood, as a door
or a vine. A bite is a crunch: a burst of cracks and a wet thump.

    select   clicking on him: a low "heh-heh-heh", a ghostly moan, or a whispered giggle and a creak
    move     ordered to move: a growl and a chuckle, or a creak and a hiss
    attack   ordered to attack: a snarl rising into a shriek, then a crunch
    train    created at the Wonder: an evil laugh, "mwa-ha-ha-ha-haaa"
    chomp    each bite of his attack animation: a crunch and a little growl
    death    a wail falling away, the crack of the pumpkin splitting, a last breath

The game plays 22050 Hz, 16-bit, mono WAV files.
"""
from __future__ import annotations

import struct

import numpy as np

SR = 22050
PEAK = 0.8

# a man's vowels (formants, Hz), lowered a little as from a big hollow head; the first is the loudest
VOWELS = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "o": (570, 840, 2410), "u": (300, 870, 2240),
          "i": (270, 2290, 3010), "m": (250, 900, 2200)}
LOWER = 0.86
WIDTHS = (90, 120, 170)
LEVELS = (1.0, 0.55, 0.3)
HOLLOW = (240, 110)  # the pumpkin's own boom: centre and width (Hz)


# --------------------------------------------------------------------------- building blocks

def _r():
    from . import roars  # roars takes SR and PEAK from here, so it is imported only when the sounds are made
    return roars


def _vowel(points, hollow: float = 0.55):
    """What his mouth lets through saying the vowels `points`, (seconds, vowel) gliding from one to the next."""
    R = _r()
    times = [p[0] for p in points]

    def gain(f, t):
        g = 0.025 + hollow * R._band(HOLLOW[0], HOLLOW[1])(f, t)
        for k in range(3):
            c = np.interp(t, times, [VOWELS[v][k] * LOWER for _, v in points])
            g = g + LEVELS[k] / np.sqrt(1 + ((f - c) / (WIDTHS[k] / 2)) ** 2)
        return g * R._highpass(60)(f, t)
    return gain


def syllable(rng, seconds: float, pitch0: float, pitch1: float, vowel: str = "a", h: float = 0.045,
             rasp: float = 0.25, breath: float = 0.3, loud: float = 1.0) -> np.ndarray:
    """"Ha" (or "heh", "hoo"): a breath, then the voice on the vowel, its pitch sliding from pitch0 to pitch1."""
    R = _r()
    t = R._time(seconds)
    pitch = np.geomspace(pitch0, pitch1, len(t))
    voice = R._voice(rng, t, pitch, rasp, 30.0, sub=0.3, breath=breath, jitter=0.035)
    on = np.clip((t - h) / 0.015, 0, 1)  # the voice starts as the "h" ends
    x = voice * on + 1.3 * R._noise(rng, t) * (1 - on)
    x = R._filter(x, _vowel([(0, vowel)]))
    env = R._env(t, [(0, 0), (0.01, 0.55), (h, 0.7), (h + 0.03, 1), (0.7 * seconds, 0.8), (seconds, 0)])
    return loud * R._grit(x, 2.0) * env


def laugh(rng, count: int = 3, pitch: float = 105.0, vowel: str = "e", seconds: float = 0.15, gap: float = 0.06,
          fall: float = 0.93) -> list[tuple[float, np.ndarray]]:
    """Syllables one after another, each a little lower and softer: placed (start, sound) for roars._mix."""
    out, at = [], 0.0
    for k in range(count):
        p = pitch * fall ** k
        out.append((at, syllable(rng, seconds, p * 1.04, p * 0.94, vowel, loud=0.92 ** k)))
        at += seconds + gap
    return out


def moan(rng, seconds: float = 1.3, pitch: float = 150.0) -> np.ndarray:
    """A ghostly "ooooOOOooo": breathy, swelling and fading, wavering, the mouth opening a little and closing."""
    R = _r()
    t = R._time(seconds)
    f0 = R._curve(t, [(0, 0.85 * pitch), (0.45 * seconds, 1.15 * pitch), (seconds, 0.8 * pitch)])
    f0 = f0 * (1 + 0.035 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.4))
    x = R._voice(rng, t, f0, 0.08, 20.0, sub=0.15, breath=0.9, jitter=0.02)
    x = R._filter(x, _vowel([(0, "u"), (0.45 * seconds, "o"), (seconds, "u")], 0.4))
    return R._norm(x) * R._env(t, [(0, 0), (0.35 * seconds, 1), (0.6 * seconds, 0.9), (seconds, 0)])


def whisper(rng, count: int = 4, vowel: str = "i", seconds: float = 0.11, gap: float = 0.05) -> np.ndarray:
    """A whispered giggle, "hihihihi": breath alone through the vowel, in quick puffs."""
    R = _r()
    total = count * (seconds + gap)
    t = R._time(total)
    puffs = np.zeros(len(t))
    for k in range(count):
        s = k * (seconds + gap)
        inside = (t >= s) & (t < s + seconds)
        u = np.clip((t - s) / seconds, 0, 1)
        puffs += inside * np.sin(np.pi * u) ** 0.6 * 0.85 ** k
    x = R._filter(R._noise(rng, t), _vowel([(0, vowel)], 0.1)) * puffs
    return R._norm(R._filter(x, R._highpass(400)))


def hiss(rng, seconds: float = 0.5) -> np.ndarray:
    """Breath hissed out through the teeth, "hhhsss"."""
    R = _r()
    t = R._time(seconds)
    x = R._filter(R._noise(rng, t), lambda f, tt: R._highpass(_r()._curve(tt, [(0, 900), (seconds, 3200)]), 2)(f, tt)
                  * R._lowpass(7000)(f, tt))
    return R._norm(x) * R._env(t, [(0, 0), (0.08, 1), (0.7 * seconds, 0.8), (seconds, 0)])


def creak(rng, seconds: float = 0.55) -> np.ndarray:
    """A slow creak, as of an old door or a twisting vine: clicks, faster then slower, ringing in wood."""
    R = _r()
    t = R._time(seconds)
    rate = R._curve(t, [(0, 16), (0.45 * seconds, 38), (seconds, 20)])
    beats = np.floor(np.cumsum(rate) / SR)
    clicks = (np.diff(beats, prepend=beats[0]) > 0).astype(float)
    wood = lambda f, tt: R._band(680, 180)(f, tt) + 0.7 * R._band(1450, 300)(f, tt) + 0.5 * R._band(330, 120)(f, tt)
    x = R._filter(clicks + 0.03 * R._noise(rng, t), wood)
    return R._norm(x) * R._env(t, [(0, 0), (0.05, 1), (0.8 * seconds, 0.85), (seconds, 0)])


def crunch(rng, seconds: float = 0.3) -> np.ndarray:
    """A bite: a burst of cracks (teeth through rind and bone) and a wet, hollow thump."""
    R = _r()
    t = R._time(seconds)
    cracks = (rng.random(len(t)) < 300 / SR) * rng.uniform(0.3, 1, len(t)) * np.exp(-t / 0.08)
    crack = R._filter(cracks + 0.35 * R._noise(rng, t) * np.exp(-t / 0.03),
                      lambda f, tt: R._highpass(800, 2)(f, tt) * R._lowpass(4500)(f, tt))
    thump = np.sin(2 * np.pi * np.cumsum(65 + 90 * np.exp(-t * 30)) / SR) * np.exp(-t * 16) * np.minimum(1, t / 0.002)
    return R._grit(R._norm(crack) + 0.9 * thump, 1.8)


def shriek(rng, seconds: float = 0.75) -> np.ndarray:
    """A monster's scream: high, rough and rising, the mouth wide on "aah" then "eeh"."""
    R = _r()
    t = R._time(seconds)
    f0 = R._curve(t, [(0, 190), (0.3 * seconds, 330), (seconds, 260)])
    x = R._voice(rng, t, f0, 0.45, 34.0, sub=0.35, breath=0.5, jitter=0.07)
    x = R._filter(x, _vowel([(0, "a"), (0.6 * seconds, "e")], 0.3))
    return R._grit(x, 3.5) * R._env(t, [(0, 0), (0.06, 1), (0.75 * seconds, 0.9), (seconds, 0)])


def wail(rng, seconds: float = 1.4) -> np.ndarray:
    """A long cry falling away, "aaaahhhh" sinking to "ooo", wavering more and more."""
    R = _r()
    t = R._time(seconds)
    f0 = np.geomspace(250, 85, len(t)) * (1 + 0.05 * np.sin(2 * np.pi * 6 * t) * np.minimum(1, t / seconds * 1.5))
    x = R._voice(rng, t, f0, 0.35, 26.0, sub=0.35, breath=0.45, jitter=0.05)
    x = R._filter(x, _vowel([(0, "a"), (0.6 * seconds, "o"), (seconds, "u")]))
    return R._grit(x, 2.5) * R._env(t, [(0, 0), (0.08, 1), (0.6 * seconds, 0.8), (seconds, 0)])


# --------------------------------------------------------------------------- the sounds

def _join(*placed, echo: float = 0.3) -> np.ndarray:
    """The sounds laid together, cleaned, echoing at `echo` (0: none, and no echo's tail, for a short bite)."""
    R = _r()
    x = R._finish(R._mix(*placed))
    if echo:
        x = R._echo(x, echo)
    return x / max(1e-9, np.abs(x).max()) * PEAK


def pacman_sounds() -> dict[str, list[np.ndarray]]:
    """Name -> variants (the game picks one at random each time)."""
    R = _r()
    r = lambda name: R._rng("pumpkin " + name)
    evil = [(0, syllable(r("mwa"), 0.42, 120, 112, "a", h=0.0, rasp=0.3))]  # "mwa": the voice from the start
    evil += [(0.5 + s, x) for s, x in laugh(r("haha"), 4, 112, "a", 0.17, 0.07, 0.95)]
    evil += [(0.5 + 4 * 0.24, syllable(r("haaa"), 0.7, 108, 72, "a", rasp=0.35))]
    return {
        "select": [_join(*laugh(r("heh"), 3, 98, "e")),
                   _join((0, moan(r("moan"))), echo=0.5),
                   _join((0, whisper(r("whisper"))), (0.62, 0.6 * creak(r("creak1"), 0.45)), echo=0.35)],
        "move": [_join((0, R.growl(r("growl"), 0.55, [(0, 70), (0.55, 85)], [(0, 0.1), (0.55, 0.3)], 24.0)),
                       (0.55, syllable(r("huh"), 0.16, 100, 90, "e"))),
                 _join((0, creak(r("creak2"))), (0.45, 0.8 * hiss(r("hiss"), 0.45)), echo=0.25)],
        "attack": [_join((0, R.growl(r("snarl"), 0.35, [(0, 75), (0.35, 110)], [(0, 0.1), (0.35, 0.5)], 28.0)),
                         (0.28, shriek(r("shriek"))), (1.0, crunch(r("bite1"))))],
        "train": [_join(*evil, echo=0.45)],
        "chomp": [_join((0, crunch(r("bite2"), 0.25)), (0.12, 0.5 * R.growl(r("gnaw"), 0.3, 80.0, 0.2, 26.0)),
                        echo=0)],
        "death": [_join((0, wail(r("wail"))), (1.25, crunch(r("split"), 0.4)), (1.3, R.thud(r("thud"))),
                        (1.6, 0.35 * hiss(r("last breath"), 0.6)), echo=0.4)],
    }


def wav(x: np.ndarray) -> bytes:
    """A 22050 Hz, 16-bit, mono PCM WAV file."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, SR, SR * 2, 2, 16)
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", len(pcm)) + pcm)
