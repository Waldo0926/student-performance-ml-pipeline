from __future__ import annotations

from typing import Sequence

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    scaler: BaseEstimator | None = None,
) -> ColumnTransformer:
    """Create leakage-safe preprocessing for numeric and categorical features."""
    if scaler is None:
        scaler = StandardScaler()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, list(numeric_features)),
            ("categorical", categorical_pipe, list(categorical_features)),
        ],
        remainder="drop",
    )


def build_svm_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    # Imputation + one-hot encoding remain inside the pipeline, but numeric scaling
    # is unnecessary for tree models.
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scaler="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_svm(
    X,
    y,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    cv_splits: int = 5,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Tune an RBF SVM using macro-F1, appropriate for the imbalanced target."""
    pipeline = build_svm_pipeline(numeric_features, categorical_features)
    param_grid = {
        "preprocessor__numeric__scaler": [
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            QuantileTransformer(output_distribution="normal", random_state=42),
        ],
        "classifier__C": [0.1, 1.0, 5.0, 10.0, 15.0, 30.0],
        "classifier__gamma": ["scale", 0.03, 0.05, 0.1, 0.2],
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    return search
