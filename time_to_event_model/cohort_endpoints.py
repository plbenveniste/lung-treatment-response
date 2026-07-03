"""
In this file we compute the cohort follow-up and recurrence patterns reported in Table 1c,
at the patient level, on the analysis cohort. Follow-up is the median observed overall-survival
time; each relapse category is counted once per patient (a patient is positive if any of the
treated lesions recurred at that site). Deaths are the patient-level counts.

Args:
    --clinical-data: Path to the raw clinical/dosimetric Excel workbook (recurrence dates)
    --patient-data: Path to the patient-level survival csv (defines the cohort and OS time)
    --output-file: Path to the json file with the table

Returns:
    None

Example:
    python model_training_review/cohort_endpoints.py --clinical-data data/clinical.xlsx --patient-data data/patient_survival.csv --output-file results/table1c.json

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import re

import pandas as pd
from loguru import logger

# Recurrence date columns mapped to the labels used in the table.
RELAPSE_SITES = {
    "Local relapse (in-field)": "Date_R_PTV",
    "Nodal relapse": "Date_R_med",
    "Homolateral lung relapse": "Date_R_homo",
    "Contralateral lung relapse": "Date_R_contro",
    "Distant metastasis": "Date_R_horspoum",
}


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Compute the cohort endpoints (Table 1c)")
    parser.add_argument("--clinical-data", type=str, help="Path to the raw clinical Excel workbook")
    parser.add_argument("--patient-data", type=str, help="Path to the patient-level survival csv")
    parser.add_argument("--output-file", type=str, help="Path to the output json")
    return parser.parse_args()


def clean_columns(columns):
    """This function normalises the noisy Excel column names."""
    cleaned = []
    for column in columns:
        name = re.sub(r"\s+", " ", str(column).replace("\n", " ").replace("\r", " ")).strip().replace(" ", "_")
        cleaned.append(re.sub(r"_+", "_", name).strip("_"))
    return cleaned


def median_range(values, decimals=1):
    """This function formats a series as median [min-max]."""
    values = values.dropna()
    return f"{values.median():.{decimals}f} [{values.min():.{decimals}f}-{values.max():.{decimals}f}]"


def count(mask, index):
    """This function formats a per-patient boolean as n (%)."""
    selected = mask.reindex(index).fillna(False)
    n = int(selected.sum())
    return f"{n} ({100 * n / len(index):.1f})"


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    patients = pd.read_csv(args.patient_data).set_index("base")
    cohort = set(patients.index)

    raw = pd.read_excel(args.clinical_data, sheet_name="Données pour tumeurs 181")
    raw.columns = clean_columns(raw.columns)
    raw["base"] = raw["num_patient"].astype(str).str.extract(r"^([A-Za-z]+\d+)")[0]
    raw = raw[raw["base"].isin(cohort)].copy()

    # We flag, per patient, whether any lesion recurred at each site.
    relapse = pd.DataFrame({
        label: raw.groupby("base")[column].apply(lambda s: s.notna().any())
        for label, column in RELAPSE_SITES.items()
        if column in raw.columns
    })
    primitive = raw.groupby("base")["primitif"].first()
    primary = primitive[primitive == 1].index
    metastatic = primitive[primitive == 2].index

    def follow_up(index):
        return median_range(patients["os_time"].reindex(index) / 365.25)

    rows = [("Follow-up time, median [range], years", follow_up(primary), follow_up(metastatic)),
            ("Deaths, n (%)", count(patients["os_event"] == 1, primary), count(patients["os_event"] == 1, metastatic))]
    for label in RELAPSE_SITES:
        if label in relapse.columns:
            rows.append((f"{label}, n (%)", count(relapse[label], primary), count(relapse[label], metastatic)))

    table = {"n_primary": int(len(primary)), "n_metastatic": int(len(metastatic)), "table_1c": rows}
    with open(args.output_file, "w") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=1)
    for label, primary_value, metastatic_value in rows:
        logger.info(f"{label:42s} {primary_value:18s} {metastatic_value}")
    logger.info(f"Saved {args.output_file}")


if __name__ == "__main__":
    main()
