from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from numbers import Number
from typing import Any, Iterable

import numpy as np
import pandas as pd


from src.types import ComparisonRules, ComparisonResult


def compare_dataframes(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    rules: ComparisonRules,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    null_string_tokens: Iterable[str] = (),
    max_unordered_unnamed_columns: int = 8,
) -> ComparisonResult:
    """
    Compare an actual DataFrame against an expected DataFrame.

    When row or column counts do not have to match, this function treats
    expected as the required subset and allows additional content in actual.
    """
    if rules.row_count_must_match and len(actual) != len(expected):
        return ComparisonResult(
            False,
            (f"Row count differs: actual={len(actual)}, expected={len(expected)}",),
        )

    if (
        rules.column_count_must_match
        and actual.shape[1] != expected.shape[1]
    ):
        return ComparisonResult(
            False,
            (
                "Column count differs: "
                f"actual={actual.shape[1]}, expected={expected.shape[1]}",
            ),
        )

    column_candidates, error = _align_column_candidates(
        actual,
        expected,
        rules,
        max_unordered_unnamed_columns,
    )

    if error:
        return ComparisonResult(False, (error,))

    null_tokens = {
        _normalize_text(
            token,
            case_matters=False,
            whitespace_matters=False,
        )
        for token in null_string_tokens
    }

    candidate_failures: list[str] = []

    for aligned_actual, aligned_expected in column_candidates:
        if (
            rules.dtypes_matter
            and list(aligned_actual.dtypes) != list(aligned_expected.dtypes)
        ):
            candidate_failures.append("Column dtypes differ")
            continue

        normalized_actual = _normalize_frame(
            aligned_actual,
            rules,
            null_tokens,
        )
        normalized_expected = _normalize_frame(
            aligned_expected,
            rules,
            null_tokens,
        )

        matches, reason = _compare_rows(
            normalized_actual,
            normalized_expected,
            rules,
            rtol=rtol,
            atol=atol,
        )

        if matches:
            return ComparisonResult(True)

        candidate_failures.append(reason)

    if not candidate_failures:
        candidate_failures.append("DataFrames differ")

    return ComparisonResult(False, (candidate_failures[0],))


def _align_column_candidates(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    rules: ComparisonRules,
    max_unordered_unnamed_columns: int,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], str | None]:
    """
    Produce possible positional column alignments.

    Normally there is exactly one alignment. Multiple candidates are needed
    only when both names and ordering are ignored.
    """
    if rules.column_names_matter:
        actual_keys = _column_occurrence_keys(actual.columns)
        expected_keys = _column_occurrence_keys(expected.columns)

        actual_positions = {
            key: position
            for position, key in enumerate(actual_keys)
        }

        missing = [
            key for key in expected_keys
            if key not in actual_positions
        ]
        if missing:
            return [], f"Expected columns are missing: {missing!r}"

        if rules.column_count_must_match and set(actual_keys) != set(expected_keys):
            return [], "Column names differ"

        positions = [
            actual_positions[key]
            for key in expected_keys
        ]

        if rules.column_order_matters:
            if rules.column_count_must_match:
                if actual_keys != expected_keys:
                    return [], "Column order differs"
            elif positions != sorted(positions):
                return [], "Expected columns appear in the wrong relative order"

        aligned_actual = actual.iloc[:, positions].copy()
        aligned_expected = expected.copy()

        # Names have already been validated, so compare positionally afterward.
        aligned_actual.columns = range(aligned_actual.shape[1])
        aligned_expected.columns = range(aligned_expected.shape[1])

        return [(aligned_actual, aligned_expected)], None

    if not rules.column_count_must_match:
        return [], (
            "Ignoring column names while allowing different column counts is "
            "ambiguous. Define how expected columns should be selected."
        )

    if rules.column_order_matters:
        aligned_actual = actual.copy()
        aligned_expected = expected.copy()

        aligned_actual.columns = range(aligned_actual.shape[1])
        aligned_expected.columns = range(aligned_expected.shape[1])

        return [(aligned_actual, aligned_expected)], None

    column_count = actual.shape[1]

    if column_count > max_unordered_unnamed_columns:
        return [], (
            "Ignoring both column names and column order requires searching "
            f"column assignments. Refusing factorial search across "
            f"{column_count} columns."
        )

    candidates: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for order in permutations(range(column_count)):
        aligned_actual = actual.iloc[:, list(order)].copy()
        aligned_expected = expected.copy()

        aligned_actual.columns = range(column_count)
        aligned_expected.columns = range(column_count)

        candidates.append((aligned_actual, aligned_expected))

    return candidates, None


def _column_occurrence_keys(
    columns: pd.Index,
) -> list[tuple[Any, int]]:
    """
    Distinguish duplicate labels such as ["a", "a"] as ("a", 0), ("a", 1).
    """
    counts: dict[Any, int] = defaultdict(int)
    keys: list[tuple[Any, int]] = []

    for label in columns:
        keys.append((label, counts[label]))
        counts[label] += 1

    return keys


def _normalize_frame(
    frame: pd.DataFrame,
    rules: ComparisonRules,
    null_tokens: set[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in result.columns:
        result[column] = result[column].map(
            lambda value: _normalize_value(
                value,
                rules,
                null_tokens,
            )
        )

    return result


def _normalize_value(
    value: Any,
    rules: ComparisonRules,
    null_tokens: set[str],
) -> Any:
    if _is_missing(value):
        return value

    if not isinstance(value, str):
        return value

    null_key = _normalize_text(
        value,
        case_matters=False,
        whitespace_matters=False,
    )

    if (
        not rules.null_representation_matters
        and null_key in null_tokens
    ):
        return pd.NA

    return _normalize_text(
        value,
        case_matters=rules.string_case_matters,
        whitespace_matters=rules.string_whitespace_matters,
    )


def _normalize_text(
    value: str,
    *,
    case_matters: bool,
    whitespace_matters: bool,
) -> str:
    if not whitespace_matters:
        # Strips edges and collapses runs of whitespace.
        value = " ".join(value.split())

    if not case_matters:
        value = value.casefold()

    return value


def _compare_rows(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    rules: ComparisonRules,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, str]:
    if rules.index_matters and actual.index.names != expected.index.names:
        return False, "Index names differ"

    actual_rows = _make_records(actual, rules.index_matters)
    expected_rows = _make_records(expected, rules.index_matters)

    if not rules.duplicate_counts_must_match:
        actual_rows = _deduplicate_records(
            actual_rows,
            rules,
            rtol=rtol,
            atol=atol,
        )
        expected_rows = _deduplicate_records(
            expected_rows,
            rules,
            rtol=rtol,
            atol=atol,
        )

    if rules.row_count_must_match and len(actual_rows) != len(expected_rows):
        return False, "Row multiplicities differ"

    if rules.row_order_matters:
        if rules.row_count_must_match:
            matches = all(
                _record_equal(
                    actual_row,
                    expected_row,
                    rules,
                    rtol=rtol,
                    atol=atol,
                )
                for actual_row, expected_row
                in zip(actual_rows, expected_rows)
            )
        else:
            matches = _expected_is_subsequence(
                actual_rows,
                expected_rows,
                rules,
                rtol=rtol,
                atol=atol,
            )
    else:
        matches = _expected_rows_have_matches(
            actual_rows,
            expected_rows,
            rules,
            rtol=rtol,
            atol=atol,
        )

    if matches:
        return True, ""

    return False, "Row values differ"


def _make_records(
    frame: pd.DataFrame,
    include_index: bool,
) -> list[tuple[Any, tuple[Any, ...]]]:
    rows = list(frame.itertuples(index=False, name=None))

    if not include_index:
        return [(None, row) for row in rows]

    return [
        (frame.index[position], row)
        for position, row in enumerate(rows)
    ]


def _deduplicate_records(
    records: list[tuple[Any, tuple[Any, ...]]],
    rules: ComparisonRules,
    *,
    rtol: float,
    atol: float,
) -> list[tuple[Any, tuple[Any, ...]]]:
    unique: list[tuple[Any, tuple[Any, ...]]] = []

    for record in records:
        already_present = any(
            _record_equal(
                record,
                existing,
                rules,
                rtol=rtol,
                atol=atol,
            )
            for existing in unique
        )

        if not already_present:
            unique.append(record)

    return unique


def _expected_is_subsequence(
    actual_rows: list[tuple[Any, tuple[Any, ...]]],
    expected_rows: list[tuple[Any, tuple[Any, ...]]],
    rules: ComparisonRules,
    *,
    rtol: float,
    atol: float,
) -> bool:
    actual_position = 0

    for expected_row in expected_rows:
        while actual_position < len(actual_rows):
            if _record_equal(
                actual_rows[actual_position],
                expected_row,
                rules,
                rtol=rtol,
                atol=atol,
            ):
                actual_position += 1
                break

            actual_position += 1
        else:
            return False

    return True


def _expected_rows_have_matches(
    actual_rows: list[tuple[Any, tuple[Any, ...]]],
    expected_rows: list[tuple[Any, tuple[Any, ...]]],
    rules: ComparisonRules,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """
    Perform bipartite matching instead of sorting.

    Sorting is unreliable for mixed types and for approximately equal numeric
    values. Matching also preserves duplicate multiplicity when required.
    """
    if len(expected_rows) > len(actual_rows):
        return False

    candidate_matches = [
        [
            actual_position
            for actual_position, actual_row in enumerate(actual_rows)
            if _record_equal(
                actual_row,
                expected_row,
                rules,
                rtol=rtol,
                atol=atol,
            )
        ]
        for expected_row in expected_rows
    ]

    matched_expected_by_actual: dict[int, int] = {}

    def assign(expected_position: int, visited: set[int]) -> bool:
        for actual_position in candidate_matches[expected_position]:
            if actual_position in visited:
                continue

            visited.add(actual_position)

            previous_expected = matched_expected_by_actual.get(actual_position)

            if previous_expected is None or assign(previous_expected, visited):
                matched_expected_by_actual[actual_position] = expected_position
                return True

        return False

    return all(
        assign(expected_position, set())
        for expected_position in range(len(expected_rows))
    )


def _record_equal(
    left: tuple[Any, tuple[Any, ...]],
    right: tuple[Any, tuple[Any, ...]],
    rules: ComparisonRules,
    *,
    rtol: float,
    atol: float,
) -> bool:
    left_index, left_values = left
    right_index, right_values = right

    if rules.index_matters and not _index_equal(
        left_index,
        right_index,
        rules.null_representation_matters,
    ):
        return False

    return all(
        _scalar_equal(
            left_value,
            right_value,
            rules,
            rtol=rtol,
            atol=atol,
        )
        for left_value, right_value
        in zip(left_values, right_values)
    )


def _scalar_equal(
    left: Any,
    right: Any,
    rules: ComparisonRules,
    *,
    rtol: float,
    atol: float,
) -> bool:
    left_missing = _is_missing(left)
    right_missing = _is_missing(right)

    if left_missing or right_missing:
        if not (left_missing and right_missing):
            return False

        if not rules.null_representation_matters:
            return True

        return _missing_kind(left) == _missing_kind(right)

    numeric = (
        isinstance(left, Number)
        and isinstance(right, Number)
        and not isinstance(left, (bool, np.bool_))
        and not isinstance(right, (bool, np.bool_))
    )

    if numeric:
        if rules.numeric_values_must_match_exactly:
            return bool(left == right)

        try:
            return bool(np.isclose(left, right, rtol=rtol, atol=atol))
        except TypeError:
            return bool(left == right)

    try:
        result = left == right
        return bool(result) if np.isscalar(result) else False
    except Exception:
        return False


def _index_equal(
    left: Any,
    right: Any,
    null_representation_matters: bool,
) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _index_equal(a, b, null_representation_matters)
            for a, b in zip(left, right)
        )

    left_missing = _is_missing(left)
    right_missing = _is_missing(right)

    if left_missing or right_missing:
        if not (left_missing and right_missing):
            return False

        return (
            not null_representation_matters
            or _missing_kind(left) == _missing_kind(right)
        )

    try:
        return bool(left == right)
    except Exception:
        return False


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except Exception:
        return False


def _missing_kind(value: Any) -> str:
    if value is None:
        return "None"
    if value is pd.NA:
        return "pd.NA"
    if value is pd.NaT:
        return "pd.NaT"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "NaN"
    if isinstance(value, (np.datetime64, np.timedelta64)) and np.isnat(value):
        return type(value).__name__

    return type(value).__name__