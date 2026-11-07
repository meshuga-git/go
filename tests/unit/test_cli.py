"""Unit tests for the shared prompt helper and the human grammar.

Requires Python 3.10 or newer, with pytest installed.
"""

from __future__ import annotations

import pytest
import unit_support  # noqa: F401 - puts the package on sys.path

import pipeline_cli


@pytest.mark.parametrize(("text", "value"), [
    ("17", 17), (" 17", 17), ("17\t", 17), ("+17", 17), ("-3", -3),
    ("017", 17), ("0", 0),
])
def test_whole_numbers_people_type(text: str, value: int) -> None:
    """Digits with a sign and surrounding spaces are whole numbers."""
    assert pipeline_cli.human_int(text) == value


@pytest.mark.parametrize("text", [
    "1_7", "１７", "١٧", "17.0", "1e1", "0x11", "", " ",
    "17 ", "- 3", "++3",
])
def test_what_int_accepts_but_people_do_not_mean(text: str) -> None:
    """Python's int() reads several of these; the human grammar not."""
    with pytest.raises(ValueError):
        pipeline_cli.human_int(text)


@pytest.mark.parametrize(("text", "value"), [
    ("0.05", 0.05), (".5", 0.5), ("5.", 5.0), ("1e-3", 0.001), ("+1", 1.0),
    (" 2 ", 2.0), ("-0.5", -0.5),
])
def test_decimal_numbers_people_type(text: str, value: float) -> None:
    """Decimals take a sign, a fraction and an exponent."""
    assert pipeline_cli.human_float(text) == value


@pytest.mark.parametrize("text", [
    "1_0", "inf", "nan", "Infinity", "1e400", "0x1p3", "", "١", "1,5",
])
def test_what_float_accepts_but_people_do_not_mean(text: str) -> None:
    """float() reads 1_0, inf and nan; the human grammar does not."""
    with pytest.raises(ValueError):
        pipeline_cli.human_float(text)


def test_sentinels_and_yes_ignore_case() -> None:
    """Exit and None match in any case; only y means yes."""
    assert pipeline_cli.is_word(" eXiT ", "Exit")
    assert not pipeline_cli.is_word("Exit now", "Exit")
    assert pipeline_cli.is_yes("Y") and pipeline_cli.is_yes(" y ")
    assert not any(pipeline_cli.is_yes(answer)
                   for answer in ("yes", "n", "", "1"))


def test_ask_explains_and_asks_again(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """A Reask prints its explanation and repeats the prompt."""
    answers = iter(["x", "1_7", "7"])
    monkeypatch.setattr("builtins.input", lambda: next(answers))

    def convert(answer: str) -> int:
        try:
            return pipeline_cli.human_int(answer)
        except ValueError:
            raise pipeline_cli.Reask("Not a whole number.") from None

    assert pipeline_cli.ask("Number: ", convert) == 7
    assert capsys.readouterr().err == (
        "Number: Not a whole number.\n" * 2 + "Number: ")
