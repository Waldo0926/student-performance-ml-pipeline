# Data

The raw course-prepared dataset is intentionally **not redistributed** in this repository.

This project expects a CSV containing at least these columns:

- `StudentID`
- `Age`
- `StudyTimeWeekly`
- `Absences`
- `ParentalSupport`
- `GPA`
- `GradeClass`

The underlying public dataset is commonly distributed as **Students Performance Dataset** on Kaggle:

https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset

The public source contains 2,392 anonymised student records and additional columns. The original university exercise used a course-prepared subset with injected data-quality issues for cleaning practice. For that reason, the exact course file is excluded here.

## Run with your own copy

Place a compatible CSV at:

```text
data/raw/student_performance.csv
```

Then run:

```bash
python scripts/run_experiment.py --data data/raw/student_performance.csv
```

`data/raw/` is git-ignored so accidental publication of private or course-distributed files is avoided.
