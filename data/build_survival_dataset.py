"""
This file builds the patient-level survival dataset used by every review-round model.
The merged, lesion-level table (clinical, dosimetric and radiomic features together with
the raw outcome dates) is collapsed to one row per patient. A patient is identified by the
letter-plus-number prefix of the lesion identifier, so that a patient treated for several
lesions is counted once. For each patient we derive three time-to-event endpoints measured
from the end of radiotherapy:
    - overall survival (OS),
    - recurrence-free survival (RFS, any local, nodal or distant recurrence, or death),
    - metastasis-free survival (MFS, contralateral, nodal or distant spread, or death).
Continuous features are averaged over the patient's lesions; the primitive/secondary label
is taken from the first lesion. The earliest treatment date is kept to allow a temporal split.

Args:
    --tumour-data: Path to the merged lesion-level csv (features + outcome dates)
    --output-file: Path to the patient-level survival csv to write

Returns:
    None

Example:
    python data/build_survival_dataset.py --tumour-data data/merged_data.csv --output-file data/patient_survival.csv

Author: Pierre-Louis Benveniste
"""

import argparse
import os

import numpy as np
import pandas as pd
from loguru import logger

# Outcome date columns expected in the merged table, and the identifier / label columns.
RECURRENCE_COLUMNS = ["Date_R_PTV", "Date_R_homo", "Date_R_med", "Date_R_contro", "Date_R_horspoum"]
DISTANT_COLUMNS = ["Date_R_horspoum", "Date_R_contro", "Date_R_med"]
NON_FEATURE_COLUMNS = {
    "num_patient", "base", "DC", "DDD", "cause_DC", "Reponse", "Last_news", "date_fin", "date_debut",
    *RECURRENCE_COLUMNS,
}


def parse_args():
    """This function is used to parse the arguments given to the script."""
    parser = argparse.ArgumentParser(description="Build the patient-level survival dataset")
    parser.add_argument("--tumour-data", type=str, help="Path to the merged lesion-level csv")
    parser.add_argument("--output-file", type=str, help="Path to the patient-level survival csv")
    return parser.parse_args()


def to_date(series):
    """This function parses a column of dates, tolerating day-first strings and blanks."""
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def patient_endpoint(base, event_date, death_date, end_of_rt, censor_date, with_death=True):
    """
    This function turns a lesion-level event date into one patient-level (time, event) pair:
    if any lesion has an event we take the earliest event time, otherwise we take the longest
    censored follow-up. Death can be added as a competing terminal event.
    """
    event = pd.concat([event_date, death_date], axis=1).min(axis=1) if with_death else event_date
    has_event = event.notna()
    time = np.where(has_event, (event - end_of_rt).dt.days, (censor_date - end_of_rt).dt.days)
    table = pd.DataFrame({"base": base, "event": has_event.astype(int), "time": np.clip(time, 1, None)})
    rows = []
    for patient, group in table.groupby("base"):
        if group["event"].max() == 1:
            rows.append((patient, 1, group.loc[group["event"] == 1, "time"].min()))
        else:
            rows.append((patient, 0, group["time"].max()))
    return pd.DataFrame(rows, columns=["base", "event", "time"]).set_index("base")


def main():
    """This is the main function of the script."""
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # We load the merged lesion-level table and derive the patient identifier.
    data = pd.read_csv(args.tumour_data)
    identifier = "num_patient" if "num_patient" in data.columns else "subject_id"
    data["base"] = data[identifier].astype(str).str.extract(r"^([A-Za-z]+\d+)")[0]

    # We parse every date we need.
    end_of_rt = to_date(data["date_fin"])
    last_news = to_date(data["Last_news"])
    death_date = to_date(data["DDD"])
    treatment_start = to_date(data["date_debut"])
    death_flag = pd.to_numeric(data["DC"], errors="coerce")
    recurrences = {c: to_date(data[c]) for c in RECURRENCE_COLUMNS}
    first_recurrence = pd.concat(recurrences.values(), axis=1).min(axis=1)
    first_distant = pd.concat([recurrences[c] for c in DISTANT_COLUMNS], axis=1).min(axis=1)
    censor_date = last_news.where(last_news.notna(), death_date)

    # We compute overall survival at the patient level.
    overall = pd.DataFrame({
        "base": data["base"],
        "event": (death_flag == 1).astype(int),
        "time": np.clip(((death_date.where(death_flag == 1, censor_date)) - end_of_rt).dt.days, 1, None),
    })
    overall = overall.groupby("base").agg(os_event=("event", "max"), os_time=("time", "first"))

    # We compute the two secondary endpoints.
    rfs = patient_endpoint(data["base"], first_recurrence, death_date, end_of_rt, censor_date)
    rfs = rfs.rename(columns={"event": "rfs_event", "time": "rfs_time"})
    mfs = patient_endpoint(data["base"], first_distant, death_date, end_of_rt, censor_date)
    mfs = mfs.rename(columns={"event": "mfs_event", "time": "mfs_time"})

    # We keep the earliest treatment date per patient (ordinal) for the temporal split.
    treat = pd.DataFrame({"base": data["base"], "start": treatment_start}).dropna()
    treat_ordinal = treat.groupby("base")["start"].min().map(lambda d: d.toordinal()).rename("treat_ord")

    # We average the numeric features over the patient's lesions and keep the primitive label.
    feature_columns = [c for c in data.columns if c not in NON_FEATURE_COLUMNS and c != "primitif"]
    aggregation = {c: "mean" for c in feature_columns}
    aggregation["primitif"] = "first"
    features = data.groupby("base").agg(aggregation)

    # We join everything and keep patients with a valid overall-survival time.
    patients = features.join([overall, rfs, mfs, treat_ordinal], how="left").reset_index()
    patients = patients[patients["os_time"] > 0].dropna(subset=["os_event", "os_time"])
    patients.to_csv(args.output_file, index=False)

    logger.info(
        f"Saved {args.output_file}: {len(patients)} patients "
        f"({int((patients.primitif == 1).sum())} primary, {int((patients.primitif != 1).sum())} metastatic)"
    )
    for endpoint in ("os", "rfs", "mfs"):
        logger.info(f"  {endpoint.upper()}: {int(patients[endpoint + '_event'].sum())} events")


if __name__ == "__main__":
    main()
