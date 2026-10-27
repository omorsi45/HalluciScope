"""Canonical text normalization.

Canonical offsets (Python `str` indices) are defined on the output of `normalize`, so every
parser routes its text through it before any span is recorded.
"""

import re
import unicodedata

_SPACES = re.compile(r" {2,}")
_TRAILING_SPACES = re.compile(r" +$", re.MULTILINE)
_NEWLINE_RUNS = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """Return the canonical form of `text`; applying it twice changes nothing.

    NFC runs after control characters are removed, not before: removing a control character
    between a base letter and a combining mark makes them adjacent, and composing only
    beforehand would leave a second pass something to compose.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch) != "Cc")
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if ch == "\t" or unicodedata.category(ch) == "Zs" else ch for ch in text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("", text)
    text = _NEWLINE_RUNS.sub("\n\n", text)
    return text.strip()
