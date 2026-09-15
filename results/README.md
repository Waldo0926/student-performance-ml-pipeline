# Results

The checked-in figures and metric tables were generated from the original 2,100-row course-prepared dataset during the portfolio refactor. The raw course file is not included.

The refactor deliberately differs from the original notebook in several ways:

- imputation is learned inside the training pipeline rather than before the split;
- model selection uses **macro-F1** rather than accuracy because the target is imbalanced;
- `StudentID` and `GPA` are excluded from predictors to avoid identifier noise and target leakage;
- categorical encoding is performed inside the sklearn pipeline;
- evaluation is reproducible from a clean process rather than relying on notebook kernel state.
