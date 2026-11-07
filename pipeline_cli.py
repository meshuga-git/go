"""Prompts and the human grammar of answers, shared by every script.

Requires Python 3.10 or newer; uses only the standard library.

Every script asks its questions through ``ask``: it shows the prompt
on stderr, reads one answer and, while the answer does not convert,
prints the script's own explanation and asks again. The scripts keep
their prompts, defaults and sentinels; this module only reads the
answers the same way everywhere.

Numbers follow one human grammar: ASCII digits with an optional sign
and an optional decimal part or exponent where a fraction is allowed,
with spaces and tabs around them ignored. So ``17``, `` 17`` and
``+17`` all mean 17, while ``1_7``, ``１７`` and ``17.0`` are not whole
numbers, and ``inf`` or ``nan`` are not numbers at all; Python's
``int()`` and ``float()`` accept them.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from typing import TypeVar

Value = TypeVar("Value")

_WHOLE_RE = re.compile(r"[+-]?[0-9]+\Z", re.ASCII)
_DECIMAL_RE = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z", re.ASCII
)


class Reask(Exception):
    """An answer to explain and ask again; the message explains it."""


def prompt_input(prompt: str = "") -> str:
    """Display an interactive prompt on stderr and read one response.

    Args:
        prompt: Prompt text; no newline is added.

    Returns:
        User response without the input line terminator.

    Raises:
        EOFError: If standard input is exhausted.
        KeyboardInterrupt: If the user interrupts input.
    """
    print(prompt, end="", file=sys.stderr, flush=True)
    return input()


def ask(prompt: str, convert: Callable[[str], Value]) -> Value:
    """Ask until an answer converts.

    Args:
        prompt: The question, printed without an added newline.
        convert: Turns the answer into its value, or raises ``Reask``
            with the explanation to print before asking again.

    Returns:
        The converted answer.

    Raises:
        EOFError: If standard input ends.
        KeyboardInterrupt: If the user interrupts input.
    """
    while True:
        answer = prompt_input(prompt)
        try:
            return convert(answer)
        except Reask as explanation:
            print(explanation, file=sys.stderr)


def _trimmed(text: str) -> str:
    """Return ``text`` without the spaces and tabs around it."""
    return text.strip(" \t")


def human_int(text: str) -> int:
    """Read a whole number as people type it.

    Args:
        text: The answer.

    Returns:
        Its value.

    Raises:
        ValueError: If the answer is not a whole number in the human
            grammar.
    """
    trimmed = _trimmed(text)
    if _WHOLE_RE.match(trimmed) is None:
        raise ValueError(f"{text!r} is not a whole number")
    return int(trimmed)


def human_float(text: str) -> float:
    """Read a decimal number as people type it.

    Args:
        text: The answer.

    Returns:
        Its value.

    Raises:
        ValueError: If the answer is not a decimal number in the human
            grammar, or too large to represent.
    """
    trimmed = _trimmed(text)
    if _DECIMAL_RE.match(trimmed) is None:
        raise ValueError(f"{text!r} is not a number")
    value = float(trimmed)
    if value in (float("inf"), float("-inf")):
        raise ValueError(f"{text!r} is too large")
    return value


def is_word(answer: str, word: str) -> bool:
    """Tell whether an answer is a sentinel word, in any case.

    Args:
        answer: The answer.
        word: The sentinel, such as ``Exit`` or ``None``.

    Returns:
        Whether the answer, without the spaces around it, is ``word``.
    """
    return answer.strip().lower() == word.lower()


def is_yes(answer: str) -> bool:
    """Tell whether an answer to a yes/no question is yes.

    Anything other than ``y``, in any case, means no.

    Args:
        answer: The answer.

    Returns:
        Whether the answer is ``y`` or ``Y``.
    """
    return answer.strip().lower() == "y"
