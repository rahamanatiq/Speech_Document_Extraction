import re
from dataclasses import dataclass
from datetime import datetime
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


# Matches a comma that sits between two digits — e.g. the "," in "12,500".
# Deliberately narrow: only strips a comma in that exact position, so a
# comma that's actually OCR noise (e.g. ",5" or a trailing "12,") is left
# alone and still correctly fails to parse, rather than being silently
# "repaired" into a number the source might not actually have said.
_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")


def _try_parse_float(text: str) -> float | None:
    cleaned = _THOUSANDS_COMMA_RE.sub("", text)
    try:
        return float(cleaned)
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
    return len(parts) == 2 and bool(parts[0].strip()) and bool(parts[1].strip())


def _split_range(text: str) -> tuple[float | None, float | None]:
    parts = text.split("-", 1)
    low = _try_parse_float(parts[0].strip())
    high = _try_parse_float(parts[1].strip())
    return low, high


# --- Unit normalisation ---
#
# Same "never guess" philosophy as normalize_value(): a unit is only ever
# rewritten into its canonical form when it confidently matches a known
# variant. Anything unrecognized is left exactly as OCR read it, flagged
# UNPARSEABLE — we never invent a canonical unit for something we don't
# recognize.


class UnitType(str, Enum):
    KNOWN = "known"              # matched a known variant, canonical form populated
    UNPARSEABLE = "unparseable"  # not a recognized unit — kept verbatim only


@dataclass
class NormalizedUnit:
    raw: str
    unit_type: UnitType
    canonical: str | None = None


# Lookup key is lowercased with whitespace stripped and "µ" folded to "u",
# so "g/dL", "G/DL", "g / dl" and "gm/dl" all resolve to the same key.
# Clinically-equivalent notations (e.g. "K/uL" and "10^3/uL" both mean
# "thousands per microlitre") are deliberately canonicalized together —
# see DECISIONS.md.
_UNIT_CANONICAL_MAP: dict[str, str] = {
    "g/dl": "g/dL",
    "gm/dl": "g/dL",
    "gdl": "g/dL",
    "mg/dl": "mg/dL",
    "mgdl": "mg/dL",
    "mmol/l": "mmol/L",
    "meq/l": "mEq/L",
    "10^3/ul": "10^3/uL",
    "10*3/ul": "10^3/uL",
    "x10^3/ul": "10^3/uL",
    "k/ul": "10^3/uL",
    "10^6/ul": "10^6/uL",
    "x10^6/ul": "10^6/uL",
    "m/ul": "10^6/uL",
    "%": "%",
    "fl": "fL",
    "pg": "pg",
    "/cmm": "/mm3",
    "cmm": "/mm3",
}


def normalize_unit(raw: str | None) -> NormalizedUnit:
    if raw is None or not raw.strip():
        return NormalizedUnit(raw="", unit_type=UnitType.UNPARSEABLE)

    text = raw.strip()
    key = text.lower().replace("µ", "u").replace(" ", "")
    canonical = _UNIT_CANONICAL_MAP.get(key)

    if canonical is not None:
        return NormalizedUnit(raw=raw, unit_type=UnitType.KNOWN, canonical=canonical)

    return NormalizedUnit(raw=raw, unit_type=UnitType.UNPARSEABLE)


# --- Date normalisation ---
#
# Same conservative approach: try a fixed list of known formats in order,
# and if none match exactly, preserve the raw string and report
# UNPARSEABLE rather than guessing. Ambiguous all-numeric formats (e.g.
# "03/04/2026" could be 3 April or 4 March) are a genuine, disclosed
# limitation — see DECISIONS.md. We only support day-first numeric slash
# format ("%d/%m/%Y"), not month-first, since that's the convention on our
# test data; a report using month-first dates will fail to parse and fall
# back to verbatim, which is the correct "never guess" behavior here.


class DateType(str, Enum):
    EXACT = "exact"
    UNPARSEABLE = "unparseable"


@dataclass
class NormalizedDate:
    raw: str
    date_type: DateType
    iso_date: str | None = None  # canonical form: YYYY-MM-DD


_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%Y/%m/%d",
]


def normalize_date(raw: str | None) -> NormalizedDate:
    if raw is None or not raw.strip():
        return NormalizedDate(raw="", date_type=DateType.UNPARSEABLE)

    text = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return NormalizedDate(raw=raw, date_type=DateType.EXACT, iso_date=parsed.strftime("%Y-%m-%d"))

    return NormalizedDate(raw=raw, date_type=DateType.UNPARSEABLE)