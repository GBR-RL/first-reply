"""English/German detection from function words.

The ticket set's own language column is unreliable: about a quarter of the tickets labelled
German are written in English. Routing results are broken down by the detected language.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-zäöüß]+")
_DE = frozenset(
    """und nicht ist ich wir sie die der das den dem des ein eine einen einem bitte mit für
    auf bei von zu wie auch sehr können könnten wurde wurden haben hat sind uns unser unsere
    ihre ihr mein meine es dass oder aber noch nach über vielen dank""".split()
)
_EN = frozenset(
    """and not is i we you they the a an to of for on with please could would have has are
    our your my it that or but still after about thank thanks this these be been was were""".split()
)


def detect(text: str) -> str:
    words = _WORD.findall(text.lower())
    de = sum(w in _DE for w in words)
    en = sum(w in _EN for w in words)
    return "de" if de > en else "en"
