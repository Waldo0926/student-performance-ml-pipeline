from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
REQUIRED_COLUMNS = {
    "StudentID",
    "Age",
    "StudyTimeWeekly",
    "Absences",
    "ParentalSupport",
    "GPA",
    "GradeClass",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV and validate the columns required by this project."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def grade_from_gpa(gpa: pd.Series) -> pd.Series:
    """Derive A/B/C/D/F labels from the documented GPA thresholds."""
    return pd.cut(
        pd.to_numeric(gpa, errors="coerce"),
        bins=[-np.inf, 2.0, 2.5, 3.0, 3.5, np.inf],
        labels=["F", "D", "C", "B", "A"],
        right=False,
    ).astype("string")


def _normalise_grade_class(series: pd.Series) -> pd.Series:
    """Accept numeric 0..4 or already-lettered labels."""
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(GRADE_MAP)
    text = series.astype("string").str.upper().str.strip()
    result = mapped.astype("string").fillna(text)
    return result


def _valid_absence_mask(absences: pd.Series, max_absences: int = 100) -> pd.Series:
    numeric = pd.to_numeric(absences, errors="coerce")
    return numeric.notna() & (numeric >= 0) & (numeric <= max_absences) & (numeric.mod(1).eq(0))


def clean_training_frame(
    frame: pd.DataFrame,
    *,
    repair_labels_from_gpa: bool = True,
    max_absences: int = 100,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply deterministic data-quality checks without learning imputation values.

    Missing feature values are intentionally left missing. Imputation belongs inside
    the sklearn pipeline so medians are learned from training folds only.
    """
    df = frame.copy()
    df["GradeClass"] = _normalise_grade_class(df["GradeClass"])

    invalid_absence = ~_valid_absence_mask(df["Absences"], max_absences=max_absences)
    invalid_absence_count = int(invalid_absence.sum())
    df = df.loc[~invalid_absence].copy()
    df["Absences"] = pd.to_numeric(df["Absences"], errors="coerce")

    gpa_grade = grade_from_gpa(df["GPA"])
    mismatch = df["GradeClass"].astype("string") != gpa_grade
    mismatch_count = int(mismatch.fillna(False).sum())

    if repair_labels_from_gpa:
        df["GradeClass"] = gpa_grade

    df = df.dropna(subset=["GradeClass"]).copy()

    audit = {
        "rows_input": int(len(frame)),
        "rows_after_absence_filter": int(len(df)),
        "invalid_absence_rows_removed": invalid_absence_count,
        "grade_label_mismatches_found": mismatch_count,
        "missing_study_time_left_for_pipeline": int(df["StudyTimeWeekly"].isna().sum()),
    }
    return df, audit


def select_features_and_target(
    frame: pd.DataFrame,
    *,
    numeric_features: Iterable[str] = ("Age", "StudyTimeWeekly", "Absences"),
    categorical_features: Iterable[str] = ("ParentalSupport",),
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Select non-leaking predictors and the GradeClass target."""
    numeric = [c for c in numeric_features if c in frame.columns]
    categorical = [c for c in categorical_features if c in frame.columns]
    features = numeric + categorical
    X = frame[features].copy()
    y = frame["GradeClass"].astype(str)
    return X, y, numeric, categorical
