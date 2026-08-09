from dataclasses import dataclass
from enum import Enum


class ValueType(str, Enum):
    """How a raw value string was interpreted, so callers know exactly
    what kind of number (if any) they're looking at."""

    EXACT = "exact"                # e.g. "13.5" -> 13.5
    LESS_THAN = "less_than"        # e.g. "<0.5" -> comparator=<, value=0.5
    GREATER_THAN = "greater_than"  # e.g. ">100" -> comparator=>, value=100
    RANGE = "range"                # e.g. "0.8 - 1.2" -> low=0.8, high=1.2
    UNPARSEABLE = "unparseable"    # couldn't confidently interpret it at all


@dataclass
class NormalizedValue:
    """The structured result of normalizing one raw value string.

    raw is always kept, no matter what — this is the same 'never discard
    the original' principle as raw_line on LabResultLine, just one level
    deeper, at the individual value level.
    """

    raw: str
    value_type: ValueType
    numeric_value: float | None = None   # populated for EXACT, LESS_THAN, GREATER_THAN
    range_low: float | None = None       # populated for RANGE
    range_high: float | None = None      # populated for RANGE


def normalize_value(raw: str | None) -> NormalizedValue:
    """Parse a raw lab-value string into structured form.

    Deliberately conservative: if the string doesn't clearly match one of
    the known patterns, we return UNPARSEABLE with numeric_value=None,
    rather than guessing. A wrong number silently entering a medical
    record is worse than an honest 'we don't know'.
    """
    if raw is None:
        return NormalizedValue(raw="", value_type=ValueType.UNPARSEABLE)

    text = raw.strip()

    # --- Range: "0.8 - 1.2" or "0.8-1.2" ---
    if _looks_like_range(text):
        low, high = _split_range(text)
        if low is not None and high is not None:
            return NormalizedValue(
                raw=raw, value_type=ValueType.RANGE, range_low=low, range_high=high
            )

    # --- Comparator: "<0.5", "< 0.5", ">100" ---
    if text.startswith("<"):
        number = _try_parse_float(text[1:].strip())
        if number is not None:
            return NormalizedValue(raw=raw, value_type=ValueType.LESS_THAN, numeric_value=number)

    if text.startswith(">"):
        number = _try_parse_float(text[1:].strip())
        if number is not None:
            return NormalizedValue(raw=raw, value_type=ValueType.GREATER_THAN, numeric_value=number)

    # --- Plain number, including scientific-ish lab notation like "1.2 x 10^3" ---
    number = _try_parse_scientific(text)
    if number is not None:
        return NormalizedValue(raw=raw, value_type=ValueType.EXACT, numeric_value=number)

    number = _try_parse_float(text)
    if number is not None:
        return NormalizedValue(raw=raw, value_type=ValueType.EXACT, numeric_value=number)

    # Nothing matched confidently — honest failure, not a guess.
    return NormalizedValue(raw=raw, value_type=ValueType.UNPARSEABLE)


def _try_parse_float(text: str) -> float | None:
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _try_parse_scientific(text: str) -> float | None:
    """Handles lab-style scientific notation: '1.2 x 10^3', '1.2 x10^3', '1.2x10^3'."""
    normalized = text.lower().replace(" ", "")
    if "x10^" not in normalized:
        return None

    base_str, _, exponent_str = normalized.partition("x10^")
    base = _try_parse_float(base_str)
    exponent = _try_parse_float(exponent_str)

    if base is None or exponent is None:
        return None
    return base * (10 ** exponent)


def _looks_like_range(text: str) -> bool:
    # Requires a hyphen with a digit on both sides — avoids false-triggering
    # on a plain negative number like "-5".
    parts = text.split("-")
    return len(parts) == 2 and parts[0].strip() and parts[1].strip()


def _split_range(text: str) -> tuple[float | None, float | None]:
    parts = text.split("-", 1)
    low = _try_parse_float(parts[0].strip())
    high = _try_parse_float(parts[1].strip())
    return low, high