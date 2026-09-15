#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from student_performance.data import clean_training_frame, load_dataset, select_features_and_target
from student_performance.evaluate import (
    classification_metrics,
    classification_report_frame,
    save_class_distribution,
    save_confusion_matrix,
    save_feature_relationships,
)
from student_performance.modeling import (
    build_random_forest_pipeline,
    build_svm_pipeline,
    tune_svm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the student-performance classification experiment.")
    parser.add_argument("--data", required=True, help="Path to a CSV containing the required columns.")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics and figures.")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip the GridSearchCV tuned SVM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(args.data)
    clean, audit = clean_training_frame(raw)
    X, y, numeric_features, categorical_features = select_features_and_target(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.70,
        stratify=y,
        random_state=66,
    )

    models = {
        "svm_baseline": build_svm_pipeline(numeric_features, categorical_features),
        "random_forest": build_random_forest_pipeline(numeric_features, categorical_features),
    }

    metric_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = {"model": name, **classification_metrics(y_test, pred)}
        metric_rows.append(row)
        classification_report_frame(y_test, pred).to_csv(output_dir / f"{name}_classification_report.csv")
        save_confusion_matrix(model, X_test, y_test, output_dir / f"{name}_confusion_matrix.png", title=name.replace("_", " ").title())

    if not args.skip_tuning:
        search = tune_svm(X_train, y_train, numeric_features, categorical_features)
        tuned = search.best_estimator_
        pred = tuned.predict(X_test)
        metric_rows.append({"model": "svm_tuned", **classification_metrics(y_test, pred)})
        classification_report_frame(y_test, pred).to_csv(output_dir / "svm_tuned_classification_report.csv")
        save_confusion_matrix(tuned, X_test, y_test, output_dir / "svm_tuned_confusion_matrix.png", title="Tuned SVM")
        pd.DataFrame([search.best_params_]).to_csv(output_dir / "svm_tuned_best_params.csv", index=False)
        (output_dir / "svm_tuned_best_cv_macro_f1.txt").write_text(f"{search.best_score_:.6f}\n", encoding="utf-8")

    pd.DataFrame(metric_rows).set_index("model").to_csv(output_dir / "metrics.csv")
    pd.DataFrame([audit]).to_csv(output_dir / "data_audit.csv", index=False)
    save_class_distribution(y, output_dir / "class_distribution.png")
    save_feature_relationships(clean, output_dir / "gpa_vs_absences.png")

    print(pd.DataFrame(metric_rows).set_index("model").round(4))
    print("\nData audit:", audit)


if __name__ == "__main__":
    main()
