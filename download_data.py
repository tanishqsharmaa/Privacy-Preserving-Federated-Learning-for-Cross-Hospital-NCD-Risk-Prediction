"""
Dataset download helper for BRFSS and NHANES.

Downloads the required XPT data files from CDC for real-world testing.
Run this script to automatically fetch all required datasets.

Usage:
    python download_data.py
"""

import os
import sys
import zipfile
import urllib.request
import shutil


# ============================================================================
# BRFSS 2022 — Behavioral Risk Factor Surveillance System
# ============================================================================
# CDC page: https://www.cdc.gov/brfss/annual_data/annual_2022.html
# 445,132 records, 326 variables

BRFSS_FILES = {
    "BRFSS 2022 (SAS Transport / XPT)": {
        "url": "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip",
        "output_dir": "data/raw/brfss",
        "filename": "LLCP2022XPT.zip",
        "size_mb": 64,
    },
}


# ============================================================================
# NHANES 2021-2023 — National Health and Nutrition Examination Survey
# ============================================================================
# CDC page: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023
# Individual XPT files, merge on SEQN variable

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"

NHANES_FILES = {
    # Demographics — age, sex, race/ethnicity, income
    "Demographics (DEMO_L)": {
        "url": f"{NHANES_BASE}/DEMO_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "DEMO_L.XPT",
        "size_mb": 2.5,
        "description": "Age (RIDAGEYR), Sex (RIAGENDR), Race/Ethnicity (RIDRETH1)",
    },
    # Body Measures — BMI, weight, height
    "Body Measures (BMX_L)": {
        "url": f"{NHANES_BASE}/BMX_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "BMX_L.XPT",
        "size_mb": 1.5,
        "description": "BMI (BMXBMI), Weight (BMXWT), Height (BMXHT)",
    },
    # Blood Pressure — systolic/diastolic readings
    "Blood Pressure (BPXO_L)": {
        "url": f"{NHANES_BASE}/BPXO_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "BPXO_L.XPT",
        "size_mb": 0.7,
        "description": "Systolic BP (BPXOSY1), Diastolic BP (BPXODI1)",
    },
    # HDL Cholesterol
    "HDL Cholesterol (HDL_L)": {
        "url": f"{NHANES_BASE}/HDL_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "HDL_L.XPT",
        "size_mb": 0.3,
        "description": "HDL Cholesterol (LBDHDD)",
    },
    # Total Cholesterol
    "Total Cholesterol (TCHOL_L)": {
        "url": f"{NHANES_BASE}/TCHOL_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "TCHOL_L.XPT",
        "size_mb": 0.3,
        "description": "Total Cholesterol (LBXTC)",
    },
    # Plasma Fasting Glucose
    "Glucose (GLU_L)": {
        "url": f"{NHANES_BASE}/GLU_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "GLU_L.XPT",
        "size_mb": 0.1,
        "description": "Fasting Glucose (LBXGLU)",
    },
    # Diabetes Questionnaire
    "Diabetes Questionnaire (DIQ_L)": {
        "url": f"{NHANES_BASE}/DIQ_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "DIQ_L.XPT",
        "size_mb": 1.2,
        "description": "Diabetes diagnosis (DIQ010)",
    },
    # Smoking Questionnaire
    "Smoking Questionnaire (SMQ_L)": {
        "url": f"{NHANES_BASE}/SMQ_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "SMQ_L.XPT",
        "size_mb": 0.5,
        "description": "Smoking status (SMQ020)",
    },
    # Alcohol Use
    "Alcohol Use (ALQ_L)": {
        "url": f"{NHANES_BASE}/ALQ_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "ALQ_L.XPT",
        "size_mb": 0.3,
        "description": "Alcohol consumption (ALQ130)",
    },
    # Physical Activity
    "Physical Activity (PAQ_L)": {
        "url": f"{NHANES_BASE}/PAQ_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "PAQ_L.XPT",
        "size_mb": 0.5,
        "description": "Physical activity (PAQ605)",
    },
    # Medical Conditions (CVD, stroke, kidney)
    "Medical Conditions (MCQ_L)": {
        "url": f"{NHANES_BASE}/MCQ_L.XPT",
        "output_dir": "data/raw/nhanes",
        "filename": "MCQ_L.XPT",
        "size_mb": 1.0,
        "description": "CVD history, stroke, kidney disease, arthritis",
    },
}


def download_file(url: str, output_path: str, description: str = ""):
    """Download a file with progress indicator."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        print(f"  [OK] Already exists: {output_path}")
        return True
    
    print(f"  >> Downloading: {description or url}")
    print(f"    -> {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  [OK] Done ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [FAIL] FAILED: {e}")
        print(f"    Please download manually from: {url}")
        return False


def extract_zip(zip_path: str, output_dir: str):
    """Extract a ZIP file."""
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print(f"  [OK] Extracted to {output_dir}")


def download_brfss():
    """Download BRFSS 2022 dataset."""
    print("\n" + "=" * 60)
    print("DOWNLOADING BRFSS 2022 (445,132 records)")
    print("Source: CDC Behavioral Risk Factor Surveillance System")
    print("=" * 60)
    
    info = BRFSS_FILES["BRFSS 2022 (SAS Transport / XPT)"]
    output_dir = info["output_dir"]
    zip_path = os.path.join(output_dir, info["filename"])
    
    success = download_file(info["url"], zip_path, "BRFSS 2022 XPT (~64 MB)")
    
    if success and zip_path.endswith(".zip"):
        extract_zip(zip_path, output_dir)
        # Check for extracted XPT file
        for f in os.listdir(output_dir):
            if f.upper().endswith(".XPT"):
                print(f"  [OK] Found data file: {os.path.join(output_dir, f)}")
    
    return success


def download_nhanes():
    """Download NHANES 2021-2023 dataset files."""
    print("\n" + "=" * 60)
    print("DOWNLOADING NHANES 2021-2023 (11 files, ~8 MB total)")
    print("Source: CDC National Health and Nutrition Examination Survey")
    print("=" * 60)
    
    all_success = True
    for name, info in NHANES_FILES.items():
        output_path = os.path.join(info["output_dir"], info["filename"])
        desc = f"{name} (~{info['size_mb']} MB) — {info.get('description', '')}"
        success = download_file(info["url"], output_path, desc)
        if not success:
            all_success = False
    
    return all_success


def verify_downloads():
    """Verify downloaded files exist and print summary."""
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    # BRFSS
    brfss_dir = "data/raw/brfss"
    brfss_files = []
    if os.path.exists(brfss_dir):
        brfss_files = [f for f in os.listdir(brfss_dir) if f.upper().endswith(".XPT")]
    print(f"\nBRFSS: {'[OK]' if brfss_files else '[X]'} {len(brfss_files)} XPT file(s) in {brfss_dir}/")
    for f in brfss_files:
        size = os.path.getsize(os.path.join(brfss_dir, f)) / (1024 * 1024)
        print(f"  - {f} ({size:.1f} MB)")
    
    # NHANES
    nhanes_dir = "data/raw/nhanes"
    nhanes_files = []
    if os.path.exists(nhanes_dir):
        nhanes_files = [f for f in os.listdir(nhanes_dir) if f.upper().endswith(".XPT")]
    print(f"\nNHANES: {'[OK]' if len(nhanes_files) >= 5 else '[X]'} {len(nhanes_files)} XPT file(s) in {nhanes_dir}/")
    for f in nhanes_files:
        size = os.path.getsize(os.path.join(nhanes_dir, f)) / (1024 * 1024)
        print(f"  - {f} ({size:.1f} MB)")
    
    if brfss_files and len(nhanes_files) >= 5:
        print("\n[OK] All datasets ready! Run the pipeline:")
        print("  python -m src.data_prep --output data/processed")
        print("  python -m src.partition --input data/processed --num-clients 10 --alpha 0.5")
    else:
        print("\n[X] Some files missing. See manual download links above.")


if __name__ == "__main__":
    print("+" + "="*50 + "+")
    print("|  PPFL-NCD Dataset Download Helper                |")
    print("|  Downloads BRFSS 2022 + NHANES 2021-2023         |")
    print("+" + "="*50 + "+")
    
    # Parse args
    download_what = "all"
    if len(sys.argv) > 1:
        download_what = sys.argv[1].lower()
    
    if download_what in ("all", "brfss"):
        download_brfss()
    
    if download_what in ("all", "nhanes"):
        download_nhanes()
    
    verify_downloads()
    
    print("\n" + "=" * 60)
    print("DONE! See above for any failures requiring manual download.")
    print("=" * 60)
