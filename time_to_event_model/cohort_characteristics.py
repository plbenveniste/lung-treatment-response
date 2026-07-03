"""
In this file we compute the patient and tumour characteristics reported in Tables 1a and 1b.
Everything is measured on the analysis cohort (the patients that entered the models) and
summarised at the patient level: patient-constant variables are read from the patient's first
lesion, tumour-level dosimetry is averaged over the lesions, matching the model input. Counts
are reported as n (%) with the cohort size as denominator and N/A for missing values;
continuous variables are reported as median [range] or mean +/- SD.

Args:
    --clinical-data: Path to the raw clinical/dosimetric Excel workbook
    --patient-data: Path to the patient-level survival csv (defines the analysis cohort)
    --output-file: Path to the json file with the two tables

Returns:
    None

Example:
    python model_training_review/cohort_characteristics.py --clinical-data data/clinical.xlsx --patient-data data/patient_survival.csv --output-file results/table1.json

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from loguru import logger

# Charlson comorbidity components (label, column) with their non-zero "present" coding.
COMORBIDITIES = [
    ("Myocardial infarction", "IDM_1"), ("Congestive heart failure", "ICC_1"),
    ("Peripheral vascular disease", "vasc_periph_1"), ("CVA or TIA", "AIT_AVC_1"),
    ("Dementia", "demence_1"), ("Chronic pulmonary disease", "BPCO_1"),
    ("Connective tissue disease", "systeme_1"), ("Peptic ulcer disease", "ulcere_1"),
    ("Mild liver disease", "hepatique_mod_1"), ("Uncomplicated diabetes", "diabete_1"),
    ("Diabetes with end-organ damage", "diabete_comp_2"), ("Hemiplegia", "hemiplegie_2"),
    ("Moderate-severe renal disease", "IRC_2"), ("Localised solid tumour", "kc_local_2"),
    ("Leukaemia", "leucemie_2"), ("Lymphoma", "lymphome_2"),
    ("Moderate-severe liver disease", "hepatique_HTP_3"), ("Metastatic solid tumour", "kc_meta_6"),
    ("AIDS", "sida_6"),
]


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Compute the cohort characteristics (Tables 1a and 1b)")
    parser.add_argument("--clinical-data", type=str, help="Path to the raw clinical Excel workbook")
    parser.add_argument("--patient-data", type=str, help="Path to the patient-level survival csv")
    parser.add_argument("--output-file", type=str, help="Path to the output json")
    return parser.parse_args()


def clean_columns(columns):
    """This function normalises the noisy Excel column names (line breaks, double spaces)."""
    cleaned = []
    for column in columns:
        name = re.sub(r"\s+", " ", str(column).replace("\n", " ").replace("\r", " ")).strip().replace(" ", "_")
        cleaned.append(re.sub(r"_+", "_", name).strip("_"))
    return cleaned


def median_range(series, index, decimals=1):
    """This function formats a continuous variable as median [min-max]."""
    values = series.reindex(index).dropna()
    if len(values) == 0:
        return "N/A"
    return f"{values.median():.{decimals}f} [{values.min():.{decimals}f}-{values.max():.{decimals}f}]"


def mean_sd(series, index, decimals=1):
    """This function formats a continuous variable as mean +/- SD."""
    values = series.reindex(index).dropna()
    if len(values) == 0:
        return "N/A"
    return f"{values.mean():.{decimals}f}\u00b1{values.std():.{decimals}f}"


def count(mask, index):
    """This function formats a boolean mask as n (%) over the cohort."""
    selected = mask.reindex(index)
    n = int(selected.sum())
    return f"{n} ({100 * n / len(index):.1f})"


def missing(series, index):
    """This function formats the number of missing values as n (%)."""
    n = int(series.reindex(index).isna().sum())
    return f"{n} ({100 * n / len(index):.1f})"


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # We restrict the raw clinical table to the analysis cohort and take one row per patient.
    cohort = set(pd.read_csv(args.patient_data)["base"])
    raw = pd.read_excel(args.clinical_data, sheet_name="Données pour tumeurs 181")
    raw.columns = clean_columns(raw.columns)
    raw = raw.rename(columns={"num_patient": "num_patient"})
    raw["base"] = raw["num_patient"].astype(str).str.extract(r"^([A-Za-z]+\d+)")[0]
    raw = raw[raw["base"].isin(cohort)].copy()
    for column in raw.columns:
        if column not in ("base", "num_patient"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    grouped = raw.groupby("base")
    first = grouped.first()             # patient-constant variables
    mean_lesion = grouped.mean(numeric_only=True)  # tumour-level dosimetry averaged per patient
    primitive = grouped["primitif"].first()
    primary = primitive[primitive == 1].index
    metastatic = primitive[primitive == 2].index
    logger.info(f"Cohort: {len(primary)} primary, {len(metastatic)} metastatic")

    def equals(series, value):
        return series == value

    # ---- Table 1a: patient characteristics ----
    table_1a = [("Age, median [range], years", median_range(first["age"], primary), median_range(first["age"], metastatic))]
    table_1a += [("Sex", " ", " "),
                 ("   Men", count(equals(first["sexe"], 0), primary), count(equals(first["sexe"], 0), metastatic)),
                 ("   Women", count(equals(first["sexe"], 1), primary), count(equals(first["sexe"], 1), metastatic)),
                 ("Performance status (WHO)", " ", " ")]
    for grade in [0, 1, 2, 3, 4]:
        table_1a.append((f"   {grade}", count(equals(first["OMS"], grade), primary), count(equals(first["OMS"], grade), metastatic)))
    table_1a.append(("   N/A", missing(first["OMS"], primary), missing(first["OMS"], metastatic)))
    table_1a += [("Smoking status", " ", " "),
                 ("   Never smokers", count(equals(first["tabac"], 0), primary), count(equals(first["tabac"], 0), metastatic)),
                 ("   Ever smokers", count(equals(first["tabac"], 1), primary), count(equals(first["tabac"], 1), metastatic)),
                 ("   of whom ceased", count(equals(first["tabac_sevre"], 1), primary), count(equals(first["tabac_sevre"], 1), metastatic)),
                 ("   N/A", missing(first["tabac"], primary), missing(first["tabac"], metastatic)),
                 ("Charlson Comorbidity Index, mean\u00b1SD", mean_sd(first["score_charlson"], primary), mean_sd(first["score_charlson"], metastatic)),
                 ("Comorbidities, n (%)", " ", " ")]
    for label, column in COMORBIDITIES:
        present = (first[column].fillna(0) != 0) if column in first.columns else pd.Series(False, index=first.index)
        table_1a.append(("   " + label, count(present, primary), count(present, metastatic)))
    table_1a += [("Hypertension", count(equals(first["HTA"], 1), primary), count(equals(first["HTA"], 1), metastatic)),
                 ("Sleep apnoea syndrome", count(equals(first["SAS"], 1), primary), count(equals(first["SAS"], 1), metastatic)),
                 ("BMI, median [range], kg/m\u00b2", median_range(first["BMI"], primary), median_range(first["BMI"], metastatic)),
                 ("Weight, median [range], kg", median_range(first["poids"], primary, 0), median_range(first["poids"], metastatic, 0))]

    # ---- Table 1b: tumour and radiotherapy characteristics ----
    table_1b = [("Height, median [range], m", median_range(first["taille"], primary, 2), median_range(first["taille"], metastatic, 2)),
                ("Dyspnoea (NYHA)", " ", " ")]
    for grade in [1, 2, 3, 4]:
        table_1b.append((f"   {grade}", count(equals(first["dyspnee_NYHA"], grade), primary), count(equals(first["dyspnee_NYHA"], grade), metastatic)))
    table_1b.append(("   N/A", missing(first["dyspnee_NYHA"], primary), missing(first["dyspnee_NYHA"], metastatic)))
    table_1b.append(("Lung tumours per patient", " ", " "))
    for count_value in [1, 2, 3]:
        table_1b.append((f"   {count_value}", count(equals(first["nbre_cibles"], count_value), primary), count(equals(first["nbre_cibles"], count_value), metastatic)))
    table_1b += [("Tumour location", " ", " "),
                 ("   Central", count(equals(first["centrale"], 1), primary), count(equals(first["centrale"], 1), metastatic)),
                 ("   Peripheral", count(equals(first["centrale"], 0), primary), count(equals(first["centrale"], 0), metastatic)),
                 ("Tumour histology", " ", " "),
                 ("   Adenocarcinoma", count(equals(first["histo"], 1), primary), count(equals(first["histo"], 1), metastatic)),
                 ("   Squamous cell", count(equals(first["histo"], 2), primary), count(equals(first["histo"], 2), metastatic)),
                 ("   Other", count(equals(first["histo"], 3), primary), count(equals(first["histo"], 3), metastatic)),
                 ("   Not histologically confirmed", count(equals(first["histo"], 4), primary), count(equals(first["histo"], 4), metastatic)),
                 ("Radiation therapy", " ", " "),
                 ("   Total dose, median [range], Gy", median_range(mean_lesion["dose_tot"], primary), median_range(mean_lesion["dose_tot"], metastatic)),
                 ("   Dose per fraction, median [range], Gy", median_range(mean_lesion["dose_fraction"], primary), median_range(mean_lesion["dose_fraction"], metastatic)),
                 ("   Overall treatment time, median [range], days", median_range(mean_lesion["etalement"], primary, 0), median_range(mean_lesion["etalement"], metastatic, 0)),
                 ("   GTV volume, median [range], cm\u00b3", median_range(mean_lesion["vol_GTV"], primary), median_range(mean_lesion["vol_GTV"], metastatic)),
                 ("   ITV volume, median [range], cm\u00b3", median_range(mean_lesion["vol_ITV"], primary), median_range(mean_lesion["vol_ITV"], metastatic)),
                 ("   PTV volume, median [range], cm\u00b3", median_range(mean_lesion["vol_PTV"], primary), median_range(mean_lesion["vol_PTV"], metastatic)),
                 ("   PTV Dmin, mean\u00b1SD, Gy", mean_sd(mean_lesion["min_PTV"], primary), mean_sd(mean_lesion["min_PTV"], metastatic)),
                 ("   PTV Dmean, mean\u00b1SD, Gy", mean_sd(mean_lesion["mean_PTV"], primary), mean_sd(mean_lesion["mean_PTV"], metastatic)),
                 ("   PTV Dmax, mean\u00b1SD, Gy", mean_sd(mean_lesion["max_PTV"], primary), mean_sd(mean_lesion["max_PTV"], metastatic)),
                 ("   BED (alpha/beta=10), median [range], Gy", median_range(mean_lesion["BED_10"], primary), median_range(mean_lesion["BED_10"], metastatic)),
                 ("   Prescription isodose, mean\u00b1SD, %", mean_sd(mean_lesion["isodose"], primary), mean_sd(mean_lesion["isodose"], metastatic)),
                 ("   PTV coverage, mean\u00b1SD, %", mean_sd(mean_lesion["couv_PTV"], primary), mean_sd(mean_lesion["couv_PTV"], metastatic))]

    tables = {"n_primary": int(len(primary)), "n_metastatic": int(len(metastatic)),
              "table_1a": table_1a, "table_1b": table_1b}
    with open(args.output_file, "w") as handle:
        json.dump(tables, handle, ensure_ascii=False, indent=1)
    logger.info(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
