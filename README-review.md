# Review-round survival analysis

This folder contains the code for the revised analysis of the lung SBRT cohort, in which
overall survival is modelled as a time-to-event outcome. It reproduces every table and figure
of the revised manuscript. The original binary-classification code is kept unchanged in
`model_training/` and `model_training_review/` from the first submission; the scripts below are
the ones used for the revision.

## Pipeline

1. **Build the analysis dataset** — collapse the merged lesion-level table to one row per
   patient and derive the three time-to-event endpoints (OS, RFS, MFS).

   ```
   python data/build_survival_dataset.py --tumour-data data/merged_data.csv --output-file data/patient_survival.csv
   ```

2. **Primary endpoint (Table 2, Figures 1-2)** — Random Survival Forest per cohort and pooled,
   with an XGBoost AFT sensitivity analysis.

   ```
   python model_training_review/overall_survival_model.py --merged-data data/patient_survival.csv --output-dir results/
   python model_training_review/plot_figures.py --results-dir results/ --output-dir results/
   ```

3. **Secondary endpoints** — recurrence-free and metastasis-free survival, cohort level only.

   ```
   python model_training_review/secondary_endpoints.py --merged-data data/patient_survival.csv --output-dir results/
   ```

4. **Interpretation (Table 3, Figure 4)** — LASSO cross-fold selection frequency and SHAP of the
   final model.

   ```
   python model_training_review/feature_selection_and_shap.py --merged-data data/patient_survival.csv --output-dir results/
   ```

5. **Incremental value (Figure 5)** — gain of the radiomic block over clinical and dosimetric
   features.

   ```
   python model_training_review/incremental_value.py --merged-data data/patient_survival.csv --output-dir results/
   ```

6. **Calibration (Figure 3)** — predicted versus observed survival at 1, 2 and 3 years.

   ```
   python model_training_review/survival_calibration.py --merged-data data/patient_survival.csv --output-dir results/
   ```

7. **Generalisation** — leave-one-centre-out and temporal split.

   ```
   python model_training_review/generalisation.py --merged-data data/patient_survival.csv --output-dir results/
   ```

8. **Cohort description (Tables 1a-1c)** — patient, tumour and outcome characteristics.

   ```
   python model_training_review/cohort_characteristics.py --clinical-data data/clinical.xlsx --patient-data data/patient_survival.csv --output-file results/table1.json
   python model_training_review/cohort_endpoints.py --clinical-data data/clinical.xlsx --patient-data data/patient_survival.csv --output-file results/table1c.json
   ```

## Notes

- Feature selection is always run with the LASSO Cox model **inside** the cross-validation
  folds; SHAP is used only to interpret the final fitted model. This separation is what keeps
  the reported concordance honest.
- All confidence intervals are percentile bootstraps (`survival_utils.bootstrap_ci`).
- `survival_utils.py` centralises preprocessing, selection, the two model builders and the
  metrics, so that every endpoint is handled identically.

## Requirements

```
pip install -r requirements-review.txt
```
