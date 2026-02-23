"""
Data preparation pipeline for BRFSS and NHANES datasets.

Handles:
- Loading raw data files (CSV / XPT)
- Feature selection and harmonization across datasets
- Categorical encoding
- Median imputation (per hospital node, not global)
- StandardScaler normalization
- SMOTE oversampling for class imbalance
- Synthetic data generation for development/testing
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger("ppfl-ncd.data_prep")


# ============================================================================
# FEATURE HARMONIZATION MAPS
# ============================================================================

# BRFSS variable name → harmonized name
# Updated for BRFSS 2022 (LLCP2022) variable naming conventions.
# Note: BRFSS 2022 does NOT have actual BP/cholesterol measurements —
#       those features will be NaN and filled via median imputation.
BRFSS_FEATURE_MAP = {
    "_AGEG5YR": "age",
    "_SEX": "sex",                              # 2022 uses _SEX (not SEX)
    "SEXVAR": "sex",                             # fallback
    "_BMI5": "bmi",
    "_IMPRACE": "race_ethnicity",               # 2022 uses _IMPRACE (not _RACE)
    "_RACE1": "race_ethnicity",                 # fallback
    "_SMOKER3": "smoking_status",
    "ALCDAY4": "alcohol_consumption",           # 2022 uses ALCDAY4 (not ALCDAY5)
    "_TOTINDA": "physical_activity",
    "DIABETE4": "diabetes",
    # Derived targets
    "_MICHD": "cardiovascular_disease",         # 1=had MI or CHD, 2=no
    # Comorbidities (available in BRFSS 2022)
    "CVDSTRK3": "has_stroke_history",           # 1=yes, 2=no
    "CHCKDNY2": "has_kidney_disease",           # 1=yes, 2=no
    "HAVARTH4": "has_arthritis",                # 1=yes, 2=no
}

# NHANES variable name → harmonized name
# Supports both pre-2021 (BPXSY1) and 2021-2023 (BPXOSY1) variable names
NHANES_FEATURE_MAP = {
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "BMXBMI": "bmi",
    "RIDRETH1": "race_ethnicity",
    "SMQ020": "smoking_status",
    "ALQ130": "alcohol_consumption",
    "PAQ605": "physical_activity",              # pre-2021
    "PAD680": "physical_activity",              # 2021-2023 (minutes sedentary)
    "BPXSY1": "blood_pressure_systolic",        # pre-2021
    "BPXDI1": "blood_pressure_diastolic",       # pre-2021
    "BPXOSY1": "blood_pressure_systolic",       # 2021-2023 (oscillometric)
    "BPXODI1": "blood_pressure_diastolic",      # 2021-2023 (oscillometric)
    "LBXTC": "cholesterol_total",
    "LBDHDD": "cholesterol_hdl",
    "LBXGLU": "blood_glucose",
    "DIQ010": "diabetes",
    # Medical Conditions Questionnaire (MCQ)
    "MCQ160B": "has_chf",                        # congestive heart failure
    "MCQ160C": "has_coronary_hd",                # coronary heart disease
    "MCQ160D": "has_angina",                     # angina
    "MCQ160E": "has_heart_attack",               # heart attack
    "MCQ160F": "has_stroke_history",             # stroke
    "KIQ022": "has_kidney_disease",              # kidney disease
    "MCQ195": "has_arthritis",                   # arthritis
}

# Harmonized feature list
HARMONIZED_FEATURES = [
    "age", "sex", "bmi", "race_ethnicity",
    "smoking_status", "alcohol_consumption", "physical_activity",
    "blood_pressure_systolic", "blood_pressure_diastolic",
    "cholesterol_total", "cholesterol_hdl", "blood_glucose",
    "has_kidney_disease", "has_stroke_history", "has_arthritis",
]

TARGET_COLUMNS = ["diabetes", "hypertension", "cardiovascular_disease"]


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_data(
    num_samples: int = 50000,
    seed: int = 42,
    source: str = "synthetic"
) -> pd.DataFrame:
    """
    Generate realistic synthetic NCD data for development/testing.
    
    Creates correlated features mimicking real-world NCD risk patterns:
    - Higher BMI → higher diabetes, hypertension, CVD risk
    - Older age → higher risk for all NCDs
    - Smoking → higher CVD risk
    - High blood pressure → hypertension label
    """
    rng = np.random.RandomState(seed)
    
    data = {
        "age": rng.randint(18, 85, num_samples).astype(float),
        "sex": rng.choice([0.0, 1.0], num_samples),  # 0=female, 1=male
        "bmi": np.clip(rng.normal(27.5, 6.0, num_samples), 15.0, 55.0),
        "race_ethnicity": rng.choice([0.0, 1.0, 2.0, 3.0, 4.0], num_samples),
        "smoking_status": rng.choice([0.0, 1.0, 2.0], num_samples),  # 0=never, 1=former, 2=current
        "alcohol_consumption": np.clip(rng.exponential(3.0, num_samples), 0, 30),
        "physical_activity": rng.choice([0.0, 1.0], num_samples),  # 0=inactive, 1=active
        "blood_pressure_systolic": np.clip(rng.normal(125, 18, num_samples), 80, 200),
        "blood_pressure_diastolic": np.clip(rng.normal(78, 12, num_samples), 50, 130),
        "cholesterol_total": np.clip(rng.normal(200, 40, num_samples), 100, 400),
        "cholesterol_hdl": np.clip(rng.normal(52, 15, num_samples), 20, 100),
        "blood_glucose": np.clip(rng.normal(100, 30, num_samples), 50, 300),
        "has_kidney_disease": rng.choice([0.0, 1.0], num_samples, p=[0.92, 0.08]),
        "has_stroke_history": rng.choice([0.0, 1.0], num_samples, p=[0.96, 0.04]),
        "has_arthritis": rng.choice([0.0, 1.0], num_samples, p=[0.75, 0.25]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate correlated targets
    # Diabetes: influenced by BMI, age, blood glucose, physical activity
    diabetes_logit = (
        -5.0
        + 0.05 * df["bmi"]
        + 0.03 * df["age"]
        + 0.02 * df["blood_glucose"]
        - 0.5 * df["physical_activity"]
        + 0.3 * df["has_kidney_disease"]
        + rng.normal(0, 0.5, num_samples)
    )
    df["diabetes"] = (1 / (1 + np.exp(-diabetes_logit)) > 0.5).astype(float)
    
    # Hypertension: influenced by blood pressure, BMI, age, alcohol
    hypertension_logit = (
        -6.0
        + 0.04 * df["blood_pressure_systolic"]
        + 0.03 * df["bmi"]
        + 0.02 * df["age"]
        + 0.05 * df["alcohol_consumption"]
        - 0.3 * df["physical_activity"]
        + rng.normal(0, 0.5, num_samples)
    )
    df["hypertension"] = (1 / (1 + np.exp(-hypertension_logit)) > 0.5).astype(float)
    
    # CVD: influenced by cholesterol, smoking, age, blood pressure, BMI
    cvd_logit = (
        -7.0
        + 0.01 * df["cholesterol_total"]
        - 0.02 * df["cholesterol_hdl"]
        + 0.5 * (df["smoking_status"] == 2).astype(float)
        + 0.04 * df["age"]
        + 0.02 * df["blood_pressure_systolic"]
        + 0.02 * df["bmi"]
        + 0.8 * df["has_stroke_history"]
        + rng.normal(0, 0.5, num_samples)
    )
    df["cardiovascular_disease"] = (1 / (1 + np.exp(-cvd_logit)) > 0.5).astype(float)
    
    df["source"] = source
    
    logger.info(f"Generated {num_samples} synthetic samples")
    logger.info(f"  Diabetes prevalence:    {df['diabetes'].mean():.1%}")
    logger.info(f"  Hypertension prevalence:{df['hypertension'].mean():.1%}")
    logger.info(f"  CVD prevalence:         {df['cardiovascular_disease'].mean():.1%}")
    
    return df


# ============================================================================
# BRFSS DATA LOADING
# ============================================================================

def load_brfss(filepath: str) -> pd.DataFrame:
    """Load and harmonize BRFSS dataset."""
    logger.info(f"Loading BRFSS data from {filepath}")
    
    # BRFSS comes as CSV or SAS format
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False)
    elif filepath.endswith(".xpt") or filepath.endswith(".XPT"):
        df = pd.read_sas(filepath, format="xport")
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Select and rename features (handle duplicate harmonized names by keeping first found)
    available_cols = {}
    seen_harmonized = set()
    for k, v in BRFSS_FEATURE_MAP.items():
        if k in df.columns and v not in seen_harmonized:
            available_cols[k] = v
            seen_harmonized.add(v)
    
    df_harmonized = df[list(available_cols.keys())].rename(columns=available_cols)
    
    # Process BMI (BRFSS stores BMI * 100)
    if "bmi" in df_harmonized.columns:
        df_harmonized["bmi"] = df_harmonized["bmi"] / 100.0
    
    # Process sex: BRFSS _SEX uses 1=Male, 2=Female → convert to 0/1
    if "sex" in df_harmonized.columns:
        df_harmonized["sex"] = (df_harmonized["sex"] == 1).astype(float)  # 1=male→1, 2=female→0
    
    # Process age: _AGEG5YR is category 1-14, convert to approximate midpoint age
    if "age" in df_harmonized.columns:
        # Categories: 1=18-24, 2=25-29, ..., 13=80+, 14=don't know
        age_midpoints = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52,
                         8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82, 14: np.nan}
        df_harmonized["age"] = df_harmonized["age"].map(age_midpoints)
    
    # Process smoking: _SMOKER3 (1=daily, 2=some days, 3=former, 4=never, 9=DK)
    if "smoking_status" in df_harmonized.columns:
        smoke_map = {1: 2.0, 2: 2.0, 3: 1.0, 4: 0.0, 9: np.nan}
        df_harmonized["smoking_status"] = df_harmonized["smoking_status"].map(smoke_map)
    
    # Process alcohol: ALCDAY4 is coded (1xx=days/week, 2xx=days/month, 888=none, 777/999=DK)
    if "alcohol_consumption" in df_harmonized.columns:
        def decode_alcday(v):
            if pd.isna(v) or v in (777, 999):
                return np.nan
            if v == 888:
                return 0.0
            v = int(v)
            if 100 <= v <= 199:
                return float(v - 100)  # days per week
            elif 200 <= v <= 299:
                return float(v - 200) / 4.3  # days per month → approx per week
            return np.nan
        df_harmonized["alcohol_consumption"] = df_harmonized["alcohol_consumption"].apply(decode_alcday)
    
    # Process physical activity: _TOTINDA (1=active, 2=inactive, 9=DK)
    if "physical_activity" in df_harmonized.columns:
        pa_map = {1: 1.0, 2: 0.0, 9: np.nan}
        df_harmonized["physical_activity"] = df_harmonized["physical_activity"].map(pa_map)
    
    # Process race: _IMPRACE (1=White, 2=Black, 3=Asian, 4=AI/AN, 5=Hispanic, 6=Other)
    if "race_ethnicity" in df_harmonized.columns:
        # Keep as numeric category (0-indexed)
        df_harmonized["race_ethnicity"] = df_harmonized["race_ethnicity"] - 1
    
    # Create binary diabetes target (DIABETE4: 1=Yes, others=No)
    if "diabetes" in df_harmonized.columns:
        df_harmonized["diabetes"] = (df_harmonized["diabetes"] == 1).astype(float)
    
    # Derive hypertension: BRFSS 2022 doesn't have BP measurements
    # Use related variables if available, else mark as NaN for imputation
    if "_RFHYPE6" in df.columns:
        # _RFHYPE6: 1=No, 2=Yes → invert
        df_harmonized["hypertension"] = (df["_RFHYPE6"] == 2).astype(float)
    else:
        # No direct hypertension variable — will be NaN, filled later
        df_harmonized["hypertension"] = np.nan
        logger.warning("BRFSS: No hypertension variable found. Will use NaN.")
    
    # CVD target: _MICHD (1=yes, 2=no)
    if "cardiovascular_disease" in df_harmonized.columns:
        df_harmonized["cardiovascular_disease"] = (
            df_harmonized["cardiovascular_disease"] == 1
        ).astype(float)
    
    # Convert comorbidity binary responses (1=Yes, 2=No, 7=DK, 9=Refused) → 0/1
    for col in ["has_kidney_disease", "has_stroke_history", "has_arthritis"]:
        if col in df_harmonized.columns:
            df_harmonized[col] = (df_harmonized[col] == 1).astype(float)
    
    # Fill missing harmonized features with NaN
    for col in HARMONIZED_FEATURES:
        if col not in df_harmonized.columns:
            df_harmonized[col] = np.nan
    
    df_harmonized["source"] = "brfss"
    logger.info(f"BRFSS: loaded {len(df_harmonized)} records with {len(available_cols)} mapped features")
    
    return df_harmonized[HARMONIZED_FEATURES + TARGET_COLUMNS + ["source"]]


# ============================================================================
# NHANES DATA LOADING
# ============================================================================

def load_nhanes(data_dir: str) -> pd.DataFrame:
    """Load and harmonize NHANES dataset (multi-file)."""
    logger.info(f"Loading NHANES data from {data_dir}")
    
    dfs = []
    for f in os.listdir(data_dir):
        fpath = os.path.join(data_dir, f)
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(fpath, low_memory=False))
        elif f.upper().endswith(".XPT"):
            dfs.append(pd.read_sas(fpath, format="xport"))
    
    if not dfs:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    # Merge on SEQN (NHANES respondent ID)
    df = dfs[0]
    for other in dfs[1:]:
        if "SEQN" in df.columns and "SEQN" in other.columns:
            df = df.merge(other, on="SEQN", how="outer", suffixes=("", "_dup"))
            # Drop duplicate columns
            df = df[[c for c in df.columns if not c.endswith("_dup")]]
    
    # Select and rename (handle duplicate harmonized names by keeping first found)
    available_cols = {}
    seen_harmonized = set()
    for k, v in NHANES_FEATURE_MAP.items():
        if k in df.columns and v not in seen_harmonized:
            available_cols[k] = v
            seen_harmonized.add(v)
    
    df_harmonized = df[list(available_cols.keys())].rename(columns=available_cols)
    
    # Process sex: RIAGENDR (1=Male, 2=Female) → 0/1
    if "sex" in df_harmonized.columns:
        df_harmonized["sex"] = (df_harmonized["sex"] == 1).astype(float)
    
    # Process smoking: SMQ020 (1=Yes ever smoked, 2=No, 7=Refused, 9=DK)
    if "smoking_status" in df_harmonized.columns:
        smoke_map = {1.0: 2.0, 2.0: 0.0, 7.0: np.nan, 9.0: np.nan}
        df_harmonized["smoking_status"] = df_harmonized["smoking_status"].map(smoke_map)
    
    # Process physical activity: PAD680 = sedentary minutes/day → invert to binary
    if "physical_activity" in df_harmonized.columns:
        pa = df_harmonized["physical_activity"]
        # Handle special codes: 7777=Refused, 9999=DK, 9999+=invalid
        pa = pa.where(pa < 1440, np.nan)  # max minutes in a day
        # Active = less than 480 min (8 hours) sedentary
        df_harmonized["physical_activity"] = (pa < 480).astype(float)
        df_harmonized.loc[pa.isna(), "physical_activity"] = np.nan
    
    # Create binary diabetes target (DIQ010: 1=Yes)
    if "diabetes" in df_harmonized.columns:
        df_harmonized["diabetes"] = (df_harmonized["diabetes"] == 1).astype(float)
    
    # Derive hypertension from blood pressure measurements
    if "blood_pressure_systolic" in df_harmonized.columns:
        bp_sys = df_harmonized["blood_pressure_systolic"]
        bp_dia = df_harmonized.get("blood_pressure_diastolic", pd.Series(dtype=float))
        hyper = (bp_sys >= 140)
        if len(bp_dia) > 0:
            hyper = hyper | (bp_dia >= 90)
        df_harmonized["hypertension"] = hyper.astype(float)
        # Keep NaN where BP is missing
        df_harmonized.loc[bp_sys.isna(), "hypertension"] = np.nan
    else:
        df_harmonized["hypertension"] = np.nan
    
    # CVD — derive from MCQ variables (1=Yes, 2=No response coding)
    if "cardiovascular_disease" not in df_harmonized.columns:
        mcq_cvd_cols = ["has_chf", "has_coronary_hd", "has_angina", "has_heart_attack"]
        has_mcq = any(c in df_harmonized.columns for c in mcq_cvd_cols)
        
        if has_mcq:
            cvd_flag = pd.Series(False, index=df_harmonized.index)
            for col in mcq_cvd_cols:
                if col in df_harmonized.columns:
                    cvd_flag = cvd_flag | (df_harmonized[col] == 1)
            df_harmonized["cardiovascular_disease"] = cvd_flag.astype(float)
        else:
            # Fallback: derive from clinical thresholds
            cvd_risk = (
                (df_harmonized.get("blood_pressure_systolic", pd.Series(0)) >= 160).astype(float) * 0.3
                + (df_harmonized.get("cholesterol_total", pd.Series(0)) >= 240).astype(float) * 0.3
                + (df_harmonized.get("smoking_status", pd.Series(0)) >= 2).astype(float) * 0.2
                + (df_harmonized.get("age", pd.Series(0)) >= 60).astype(float) * 0.2
            )
            df_harmonized["cardiovascular_disease"] = (cvd_risk >= 0.5).astype(float)
    
    # Convert MCQ binary responses (1=Yes, 2=No) to 0/1
    for col in ["has_kidney_disease", "has_stroke_history", "has_arthritis"]:
        if col in df_harmonized.columns:
            df_harmonized[col] = (df_harmonized[col] == 1).astype(float)
    
    # Fill missing
    for col in HARMONIZED_FEATURES:
        if col not in df_harmonized.columns:
            df_harmonized[col] = np.nan
    
    df_harmonized["source"] = "nhanes"
    logger.info(f"NHANES: loaded {len(df_harmonized)} records with {len(available_cols)} mapped features")
    
    return df_harmonized[HARMONIZED_FEATURES + TARGET_COLUMNS + ["source"]]


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess(
    df: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess a harmonized DataFrame.
    
    Steps:
    1. Median imputation for missing values
    2. StandardScaler normalization on continuous features
    
    Args:
        df: Harmonized DataFrame with features + targets + source
        fit_scaler: If True, fit a new scaler; if False, use provided scaler
        scaler: Pre-fitted scaler for transform-only mode
    
    Returns:
        (processed_df, fitted_scaler)
    """
    df = df.copy()
    
    # Continuous features to scale
    continuous_cols = [
        "age", "bmi", "alcohol_consumption",
        "blood_pressure_systolic", "blood_pressure_diastolic",
        "cholesterol_total", "cholesterol_hdl", "blood_glucose"
    ]
    
    # Categorical (already encoded as numeric)
    categorical_cols = [
        "sex", "race_ethnicity", "smoking_status", "physical_activity",
        "has_kidney_disease", "has_stroke_history", "has_arthritis"
    ]
    
    # Step 1: Median imputation
    for col in HARMONIZED_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Step 2: Scale continuous features
    available_continuous = [c for c in continuous_cols if c in df.columns]
    
    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        df[available_continuous] = scaler.fit_transform(df[available_continuous])
    else:
        df[available_continuous] = scaler.transform(df[available_continuous])
    
    # Ensure targets are binary float, fill NaN targets with 0 (negative class)
    # This handles cases like BRFSS 2022 having no hypertension variable
    for col in TARGET_COLUMNS:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"Target '{col}' has {nan_count} NaN values "
                    f"({nan_count/len(df):.1%}). Filling with 0 (negative class)."
                )
                df[col] = df[col].fillna(0.0)
            df[col] = df[col].astype(float)
    
    return df, scaler


def prepare_dataset(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    use_synthetic: bool = False,
    synthetic_samples: int = 50000,
    seed: int = 42,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full data preparation pipeline.
    
    Args:
        raw_dir: Directory containing raw BRFSS/NHANES files
        output_dir: Directory to save processed data
        use_synthetic: If True, generate synthetic data instead of loading real data
        synthetic_samples: Number of synthetic samples
        seed: Random seed
        test_size: Fraction for test set
    
    Returns:
        (train_df, test_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if use_synthetic:
        logger.info("Using synthetic data generation")
        # Generate two "sources" to simulate BRFSS + NHANES
        df_brfss = generate_synthetic_data(
            synthetic_samples // 2, seed=seed, source="brfss"
        )
        df_nhanes = generate_synthetic_data(
            synthetic_samples // 2, seed=seed + 1, source="nhanes"
        )
        # Make NHANES slightly different (non-IID simulation)
        df_nhanes["bmi"] += np.random.RandomState(seed).normal(2, 1, len(df_nhanes))
        df_nhanes["age"] += np.random.RandomState(seed).normal(5, 2, len(df_nhanes))
        df_nhanes["age"] = np.clip(df_nhanes["age"], 18, 90)
        
        df = pd.concat([df_brfss, df_nhanes], ignore_index=True)
    else:
        # Load real datasets
        dfs = []
        
        brfss_path = os.path.join(raw_dir, "brfss")
        if os.path.exists(brfss_path):
            for f in os.listdir(brfss_path):
                if f.endswith((".csv", ".XPT", ".xpt")):
                    try:
                        dfs.append(load_brfss(os.path.join(brfss_path, f)))
                    except Exception as e:
                        logger.warning(f"Failed to load BRFSS file {f}: {e}")
        
        nhanes_path = os.path.join(raw_dir, "nhanes")
        if os.path.exists(nhanes_path):
            try:
                dfs.append(load_nhanes(nhanes_path))
            except Exception as e:
                logger.warning(f"Failed to load NHANES: {e}")
        
        if not dfs:
            logger.warning("No real data found. Falling back to synthetic data.")
            return prepare_dataset(
                raw_dir, output_dir, use_synthetic=True,
                synthetic_samples=synthetic_samples, seed=seed, test_size=test_size
            )
        
        df = pd.concat(dfs, ignore_index=True)
    
    # Train/test split (stratified by source)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["source"]
    )
    
    # Preprocess (fit scaler on train only)
    train_df, scaler = preprocess(train_df, fit_scaler=True)
    test_df, _ = preprocess(test_df, fit_scaler=False, scaler=scaler)
    
    # Save
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Save scaler for later use
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    
    logger.info(f"Saved {len(train_df)} train, {len(test_df)} test samples to {output_dir}")
    logger.info(f"Features: {HARMONIZED_FEATURES}")
    logger.info(f"Targets: {TARGET_COLUMNS}")
    
    # Print statistics
    for col in TARGET_COLUMNS:
        logger.info(f"  {col} train prevalence: {train_df[col].mean():.1%}")
        logger.info(f"  {col} test prevalence:  {test_df[col].mean():.1%}")
    
    return train_df, test_df


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NCD dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data instead of loading real data")
    parser.add_argument("--synthetic-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    
    args = parser.parse_args()
    
    from src.utils import setup_logging
    setup_logging()
    
    prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output,
        use_synthetic=args.synthetic,
        synthetic_samples=args.synthetic_samples,
        seed=args.seed,
        test_size=args.test_size,
    )
