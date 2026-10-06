from __future__ import annotations

import random


ADJECTIVES = [
    "Tremendous",
    "Lonely",
    "Brisk",
    "Curious",
    "Faint",
    "Mellow",
    "Grand",
    "Nimble",
    "Quiet",
    "Bright",
    "Solar",
    "Rustic",
    "Candid",
    "Eager",
    "Humble",
    "Lively",
    "Noble",
    "Swift",
    "Vivid",
    "Witty",
]

NOUNS = [
    "Pig",
    "Heron",
    "Otter",
    "Falcon",
    "Bison",
    "Lynx",
    "Badger",
    "Dolphin",
    "Puma",
    "Gull",
    "Hawk",
    "Fox",
    "Salmon",
    "Orca",
    "Rabbit",
    "Marten",
    "Coyote",
    "Turtle",
    "Robin",
    "Sparrow",
]


def generate_codename() -> str:
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adjective}{noun}"
