from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

LABEL_ORDER = ["A", "B", "C", "D", "F"]


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def classification_report_frame(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def save_confusion_matrix(model, X_test, y_test, path: str | Path, *, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        labels=LABEL_ORDER,
        values_format="d",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_class_distribution(y: pd.Series, path: str | Path) -> None:
    path = Path(path)
    counts = y.value_counts().reindex(LABEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Grade class distribution")
    ax.set_xlabel("Grade class")
    ax.set_ylabel("Students")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_feature_relationships(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    x = pd.to_numeric(frame["Absences"], errors="coerce")
    y = pd.to_numeric(frame["GPA"], errors="coerce")
    mask = x.notna() & y.notna()
    ax.scatter(x[mask], y[mask], s=12, alpha=0.55)
    if mask.sum() >= 2:
        m, b = np.polyfit(x[mask], y[mask], 1)
        xx = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xx, m * xx + b)
    corr = y.corr(x)
    ax.set_title(f"GPA vs absences (Pearson r = {corr:.3f})")
    ax.set_xlabel("Absences")
    ax.set_ylabel("GPA")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
