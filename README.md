Project: Bat–Rat Interaction Analysis (HIT140 Assessment 2)

Overview

- Purpose: Clean two field datasets, explore/visualize key variables, and run one-sample and two-sample hypothesis tests to understand relationships between rat presence/behavior and bat activity.
- Artifacts: Jupyter notebooks for cleaning and analysis, resulting cleaned CSVs, and figures.

Repository Structure

- `Final_files/dataset_1_cleaning.ipynb`: Cleaning + EDA for dataset 1; prepares variables, fixes categorical noise in `habit`, handles time parsing, IQR-based outlier handling, and visualizations.
- `Final_files/dataset_2_cleaning.ipynb`: Cleaning + EDA for dataset 2; schema checks, outlier inspection, histograms, and optional save of a cleaned version.
- `Final_files/One_Sample_Hypothesis_Testing.ipynb`: One-sample tests on cleaned dataset 1 (binomial tests, z-test, t-test).
- `Final_files/Two_Sample_Hypothesis_Testing.ipynb`: Two-sample comparisons on cleaned dataset 2 split by rat presence metrics.
- `Datasets/dataset1.csv`: Raw dataset 1.
- `Datasets/dataset2.csv`: Raw dataset 2.
- `Datasets/cleaned_dataset1.csv`: Cleaned dataset 1 output used by analysis.
- `Datasets/dataset2_cleaned.csv`, `Datasets/dataset2_cleaned_V2.csv`: Cleaned dataset 2 outputs used by analysis.
- `Final_files/correlation.png`: Saved correlation heatmap image (from EDA).

Data Summary

- Dataset 1 columns (core): `start_time`, `bat_landing_to_food`, `habit`, `rat_period_start`, `rat_period_end`, `seconds_after_rat_arrival`, `risk`, `reward`, `month` (plus derived time parts from parsing and normalizing).
- Dataset 2 columns: `time`, `month`, `hours_after_sunset`, `bat_landing_number`, `food_availability`, `rat_minutes`, `rat_arrival_number`.

Notebooks: What They Do

- `Final_files/dataset_1_cleaning.ipynb`
  - Reads raw `Datasets/dataset1.csv` (and in places also references dataset 2 for comparison).
  - Cleans `habit` by consolidating noisy/mis-encoded labels, and drops a small number of nulls in `habit` where appropriate.
  - Parses date/time strings (`start_time`, `rat_period_start`, `rat_period_end`) and derives hour parts for analysis.
  - Converts dtypes for categorical-like fields (`risk`, `reward`, `month`, `season` mentioned in code comments).
  - Outlier handling: IQR-based identification/removal is applied for `bat_landing_to_food` (shows counts and percentages), with before/after boxplots.
  - EDA: histograms, boxplots, and a correlation heatmap (saved as `Final_files/correlation.png`).
  - Produces a cleaned CSV used downstream: `Datasets/cleaned_dataset1.csv`.

- `Final_files/dataset_2_cleaning.ipynb`
  - Reads `Datasets/dataset2.csv` and prints schema/shape.
  - Visualizes distributions (histograms) for numeric columns.
  - Uses IQR-based ranges to inspect potential outliers per numeric column; documents counts/percentages.
  - Rationale: keeps certain outliers to retain useful signal (notes that removing all would overly reduce information).
  - Optionally saves `dataset2_cleaned_V2.csv` (present as `Datasets/dataset2_cleaned_V2.csv`).

- `Final_files/One_Sample_Hypothesis_Testing.ipynb`
  - Loads `cleaned_dataset1.csv`.
  - Binomial tests:
    - `risk` column: tests whether the count of zeros differs from p=0.5; p≈0.4713 → fail to reject H0.
    - `reward` column: tests proportion of zeros; prints counts/p-value; conclusion: fail to reject H0.
    - Derived `fear_column` from `habit` (mapping several rat/bat interaction patterns to 1=fear, else 0); binomial test p≈1.0 → fail to reject H0.
  - One-sample tests:
    - Z-test on `hours_after_sunset` (sample of n=100 vs population mean): prints Z and p, with alpha=0.05 decision.
    - One-sided t-test on `bat_landing_to_food` vs threshold 7: sample mean ≈5.97, p≈0.99999 → fail to reject H0.

- `Final_files/Two_Sample_Hypothesis_Testing.ipynb`
  - Loads `dataset2_cleaned_V2.csv`.
  - Splitting strategy: create two groups per split variable
    - Group A: rows where `rat_arrival_number` > 0 (or `rat_minutes` > 0)
    - Group B: rows where the split variable equals 0
  - For each target column (`bat_landing_number`, `food_availability`, `hours_after_sunset`):
    - Computes group stats (mean, std, n) and runs a two-sample z-test.
    - Results consistently show very small p-values → reject H0 (group means differ) across both split variables.

Environment Setup

- Python: 3.9+ recommended.
- Install packages:
  - `pandas`, `numpy`, `matplotlib`, `plotly`, `scipy`, `statsmodels`, `jupyter`.
- Example (pip):
  - `python -m venv .venv && source .venv/bin/activate` (Windows PowerShell: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`)
  - `pip install pandas numpy matplotlib plotly scipy statsmodels jupyter`

How To Run

- Open notebooks: `jupyter lab` or `jupyter notebook` from repository root.
- Cleaning:
  - Run `Final_files/dataset_1_cleaning.ipynb` to reproduce EDA and cleaning for dataset 1. It should produce/confirm `Datasets/cleaned_dataset1.csv`.
  - Run `Final_files/dataset_2_cleaning.ipynb` for EDA and (optionally) save `Datasets/dataset2_cleaned_V2.csv`.
- Analysis:
  - Run `Final_files/One_Sample_Hypothesis_Testing.ipynb` (expects `cleaned_dataset1.csv` present in working directory; if needed, change the read path to `Datasets/cleaned_dataset1.csv`).
  - Run `Final_files/Two_Sample_Hypothesis_Testing.ipynb` (expects `dataset2_cleaned_V2.csv`; if needed, change the read path to `Datasets/dataset2_cleaned_V2.csv`).

Path Notes & Tips

- Some cells in `Final_files/dataset_2_cleaning.ipynb` use an absolute Windows path when reading/writing CSVs. If you encounter a FileNotFoundError, replace those with relative paths, for example:
  - Read raw: `pd.read_csv('Datasets/dataset2.csv')`
  - Save cleaned: `df_cleaned.to_csv('Datasets/dataset2_cleaned_V2.csv', index=False)`
- The hypothesis notebooks sometimes read `cleaned_dataset1.csv` or `dataset2_cleaned_V2.csv` from the current working directory. If running from repo root, prepend `Datasets/` to these filenames.

Key Results (High Level)

- Dataset 1 (one-sample):
  - Proportions in `risk`, `reward`, and `fear_column` do not significantly deviate from p=0.5 in one-sided binomial tests (fail to reject H0).
  - `hours_after_sunset` sample mean vs population mean: z-test reports Z and p; decision shown per run (alpha=0.05).
  - `bat_landing_to_food` mean vs 7: one-sided t-test yields p≈0.99999 (fail to reject H0).
- Dataset 2 (two-sample):
  - When splitting by rat presence (`rat_arrival_number` > 0 vs = 0; `rat_minutes` > 0 vs = 0), group means for `bat_landing_number`, `food_availability`, and `hours_after_sunset` differ with very small p-values (reject H0 consistently).

Reproducibility

- All necessary cleaned CSVs are included in `Datasets/` as checked-in artifacts. You can rerun cleaning notebooks to regenerate them or run analyses directly.
- Because some tests involve random sampling (e.g., z-test on a sample of 100), the exact numeric results (Z/p) can vary between runs unless you keep the provided `random_state`.

Troubleshooting

- File paths: If a notebook cannot find a CSV, verify the working directory and use the `Datasets/` prefix in file paths.
- Plotly rendering: In some environments, interactive Plotly figures may require JupyterLab/Notebook extensions. If interactive plots don’t render, fall back to static Matplotlib plots or export figures with `fig.write_image`.
- Package versions: If you see unexpected numerical or plotting differences, confirm your package versions and consider pinning a set in a `requirements.txt`.

