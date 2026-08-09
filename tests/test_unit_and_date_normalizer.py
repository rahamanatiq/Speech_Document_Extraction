import pytest

from services.normalizer import (
    DateType,
    UnitType,
    ValueType,
    normalize_date,
    normalize_unit,
    normalize_value,
)


class TestCommaThousandsSeparator:
    def test_comma_separated_integer(self):
        result = normalize_value("12,500")
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == 12500.0

    def test_comma_separated_decimal(self):
        result = normalize_value("1,234.5")
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == 1234.5

    def test_trailing_comma_not_silently_repaired(self):
        # A comma NOT sandwiched between two digits is not a thousands
        # separator — this must still fail honestly, not get "fixed" into
        # a number that might not be what was actually written.
        result = normalize_value("12,")
        assert result.value_type == ValueType.UNPARSEABLE


class TestUnitNormalization:
    @pytest.mark.parametrize(
        "raw,expected_canonical",
        [
            ("g/dL", "g/dL"),
            ("gm/dl", "g/dL"),
            ("G/DL", "g/dL"),
            ("mg/dl", "mg/dL"),
            ("mmol/L", "mmol/L"),
            ("K/µL", "10^3/uL"),
            ("10^3/uL", "10^3/uL"),
            ("x10^3/ul", "10^3/uL"),
        ],
    )
    def test_known_variants_map_to_canonical_form(self, raw, expected_canonical):
        result = normalize_unit(raw)
        assert result.unit_type == UnitType.KNOWN
        assert result.canonical == expected_canonical

    def test_unrecognized_unit_preserved_verbatim(self):
        result = normalize_unit("blorp/xyz")
        assert result.unit_type == UnitType.UNPARSEABLE
        assert result.canonical is None
        assert result.raw == "blorp/xyz"

    def test_none_input(self):
        result = normalize_unit(None)
        assert result.unit_type == UnitType.UNPARSEABLE

    def test_raw_always_preserved(self):
        result = normalize_unit("gm/dl")
        assert result.raw == "gm/dl"  # original spelling kept even though canonical differs


class TestDateNormalization:
    @pytest.mark.parametrize(
        "raw,expected_iso",
        [
            ("2026-03-14", "2026-03-14"),
            ("14-03-2026", "2026-03-14"),
            ("14/03/2026", "2026-03-14"),
            ("14 Mar 2026", "2026-03-14"),
            ("14 March 2026", "2026-03-14"),
            ("Mar 14, 2026", "2026-03-14"),
        ],
    )
    def test_known_formats_normalize_to_iso(self, raw, expected_iso):
        result = normalize_date(raw)
        assert result.date_type == DateType.EXACT
        assert result.iso_date == expected_iso

    def test_unparseable_date_preserved_verbatim(self):
        result = normalize_date("sometime last spring")
        assert result.date_type == DateType.UNPARSEABLE
        assert result.iso_date is None
        assert result.raw == "sometime last spring"

    def test_none_input(self):
        result = normalize_date(None)
        assert result.date_type == DateType.UNPARSEABLE

    def test_empty_string(self):
        result = normalize_date("")
        assert result.date_type == DateType.UNPARSEABLE