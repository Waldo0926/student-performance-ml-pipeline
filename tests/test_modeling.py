import numpy as np
import pandas as pd

from student_performance.modeling import build_svm_pipeline


def test_svm_pipeline_handles_missing_values_and_unseen_category():
    X = pd.DataFrame(
        {
            "Age": [15, 16, 17, 18] * 5,
            "StudyTimeWeekly": [2.0, 6.0, np.nan, 14.0] * 5,
            "Absences": [25, 15, 6, 1] * 5,
            "ParentalSupport": [0, 1, 2, 3] * 5,
        }
    )
    y = pd.Series(["F", "D", "C", "A"] * 5)
    model = build_svm_pipeline(
        ["Age", "StudyTimeWeekly", "Absences"],
        ["ParentalSupport"],
    )
    model.fit(X, y)
    pred = model.predict(
        pd.DataFrame(
            {
                "Age": [17],
                "StudyTimeWeekly": [8.0],
                "Absences": [7],
                "ParentalSupport": [4],
            }
        )
    )
    assert pred.shape == (1,)
