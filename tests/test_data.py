import numpy as np
import pandas as pd

from student_performance.data import clean_training_frame, grade_from_gpa, select_features_and_target


def test_grade_from_gpa_boundaries():
    values = pd.Series([3.5, 3.0, 2.5, 2.0, 1.99])
    assert grade_from_gpa(values).tolist() == ["A", "B", "C", "D", "F"]


def test_cleaning_filters_invalid_absences_and_defers_imputation():
    df = pd.DataFrame(
        {
            "StudentID": [1, 2, 3],
            "Age": [16, 17, 18],
            "StudyTimeWeekly": [np.nan, 6.0, 8.0],
            "Absences": [3, -1, 4],
            "ParentalSupport": [1, 2, 3],
            "GPA": [3.6, 2.7, 1.5],
            "GradeClass": [0, 2, 4],
        }
    )
    clean, audit = clean_training_frame(df)
    assert len(clean) == 2
    assert audit["invalid_absence_rows_removed"] == 1
    assert clean["StudyTimeWeekly"].isna().sum() == 1
    assert audit["missing_study_time_left_for_pipeline"] == 1


def test_feature_selection_excludes_id_and_gpa():
    df = pd.DataFrame(
        {
            "StudentID": [1],
            "Age": [16],
            "StudyTimeWeekly": [5.0],
            "Absences": [3],
            "ParentalSupport": [2],
            "GPA": [3.6],
            "GradeClass": ["A"],
        }
    )
    X, y, numeric, categorical = select_features_and_target(df)
    assert "StudentID" not in X.columns
    assert "GPA" not in X.columns
    assert numeric == ["Age", "StudyTimeWeekly", "Absences"]
    assert categorical == ["ParentalSupport"]
    assert y.tolist() == ["A"]
