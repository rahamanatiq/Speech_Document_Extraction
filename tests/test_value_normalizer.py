import pytest

from services.normalizer import ValueType, normalize_value


class TestExactValues:
    def test_plain_integer(self):
        result = normalize_value("13")
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == 13.0

    def test_plain_decimal(self):
        result = normalize_value("13.5")
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == 13.5

    def test_negative_number_not_treated_as_range(self):
        # The exact bug class this test guards against: a naive "-" in text
        # check would wrongly treat "-5" as a broken range (low=None, high=5).
        result = normalize_value("-5")
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == -5.0

    def test_whitespace_is_stripped(self):
        result = normalize_value("   14.2   ")
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == 14.2

    @pytest.mark.parametrize("raw", ["1.2 x 10^3", "1.2x10^3", "1.2 X 10^3"])
    def test_scientific_notation_variants(self, raw):
        result = normalize_value(raw)
        assert result.value_type == ValueType.EXACT
        assert result.numeric_value == 1200.0


class TestComparators:
    def test_less_than(self):
        result = normalize_value("<0.5")
        assert result.value_type == ValueType.LESS_THAN
        assert result.numeric_value == 0.5

    def test_less_than_with_space(self):
        result = normalize_value("< 0.5")
        assert result.value_type == ValueType.LESS_THAN
        assert result.numeric_value == 0.5

    def test_greater_than(self):
        result = normalize_value(">100")
        assert result.value_type == ValueType.GREATER_THAN
        assert result.numeric_value == 100.0


class TestRanges:
    def test_range_with_spaces(self):
        result = normalize_value("0.8 - 1.2")
        assert result.value_type == ValueType.RANGE
        assert result.range_low == 0.8
        assert result.range_high == 1.2

    def test_range_without_spaces(self):
        result = normalize_value("0.8-1.2")
        assert result.value_type == ValueType.RANGE
        assert result.range_low == 0.8
        assert result.range_high == 1.2


class TestUnparseable:
    def test_ocr_garbled_text(self):
        # The core "never guess" case: OCR misread "12.5" as "l2.S".
        # This must NEVER silently resolve to a number.
        result = normalize_value("l2.S")
        assert result.value_type == ValueType.UNPARSEABLE
        assert result.numeric_value is None

    def test_empty_string(self):
        result = normalize_value("")
        assert result.value_type == ValueType.UNPARSEABLE

    def test_none_input(self):
        result = normalize_value(None)
        assert result.value_type == ValueType.UNPARSEABLE

    def test_pure_garbage(self):
        result = normalize_value("???###")
        assert result.value_type == ValueType.UNPARSEABLE


class TestRawIsAlwaysPreserved:
    """The single most important property of this module: no matter what
    happens during parsing, `raw` always reflects the original input.
    """

    @pytest.mark.parametrize(
        "raw",
        ["13.5", "<0.5", ">100", "0.8-1.2", "1.2x10^3", "l2.S", "garbage"],
    )
    def test_raw_matches_input_exactly(self, raw):
        result = normalize_value(raw)
        assert result.raw == raw