"""Utilities for handling feature exclusion patterns."""

from fnmatch import fnmatch
from typing import Iterable, Set, Tuple

import pandas as pd


def resolve_ignored_columns(columns: Iterable[str], patterns: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """Resolve wildcard patterns against column names."""
    matched_columns: Set[str] = set()
    matched_patterns: Set[str] = set()

    for name in columns:
        for pattern in patterns:
            if fnmatch(name, pattern):
                matched_columns.add(name)
                matched_patterns.add(pattern)
                break

    unmatched_patterns = set(patterns) - matched_patterns
    return matched_columns, unmatched_patterns


def mask_ignored_features(df: pd.DataFrame, patterns: Iterable[str]) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """Return a copy of *df* with ignored columns set to zero."""
    ignored_columns, unmatched_patterns = resolve_ignored_columns(df.columns, patterns)
    if not ignored_columns:
        return df, ignored_columns, unmatched_patterns

    masked = df.copy()
    masked.loc[:, list(ignored_columns)] = 0.0
    return masked, ignored_columns, unmatched_patterns
