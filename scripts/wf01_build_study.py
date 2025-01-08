from classes import xmap_study, Normalizer
import pandas as pd
import pickle


# Step 1: Initialize the study
def initialize_study(study_file, control_columns):
    return xmap_study(study_report_xls=study_file, control_cols=control_columns)


# Step 2: Attach patient data and clean column names
def attach_and_clean_patient_data(study, patient_data_file):
    patient_data = pd.read_excel(patient_data_file, index_col=0)
    study.attach_patient_data(patient_data)
    study.adj_mfi_data.columns = study.adj_mfi_data.columns.str.replace(
        "*", "", regex=False
    )
    study.adj_mfi_data.columns = study.adj_mfi_data.columns.str.replace(
        "-", "_", regex=False
    )
    return study


# Step 3: Normalize the data
def normalize_data(study):
    normalizer = Normalizer()
    normalized_data = normalizer.normalize(
        study.adj_mfi_data.drop(columns=["Patient_ID"]),
        rsn_transform=True,
        boxcox_transform=True,
    )
    study.normalized_data = normalized_data
    return study


# Save the study object to avoid rerunning steps 1-3
def save_study(study, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(study, f)


# Main execution
def main():
    study = initialize_study("data/02_assay_data.xlsx", control_columns=slice(-4, None))
    study = attach_and_clean_patient_data(study, "data/02_patient_data_cleaned.xlsx")
    study = normalize_data(study)
    save_study(study, "data/study.pickle")


if __name__ == "__main__":
    main()
