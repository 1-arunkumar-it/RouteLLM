"""Representation of a single detected keyword or phrase signal."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """A matched keyword or phrase and the category it supports."""

    phrase: str
    category: str
