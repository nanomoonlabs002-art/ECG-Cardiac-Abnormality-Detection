"""
ECG Heart Disease Detection
STEP 1 - Download MIT-BIH Dataset
Run: python download_data.py
"""
import wfdb
import os

print("=" * 50)
print("  Downloading ECG Dataset...")
print("=" * 50)

os.makedirs("mitbih_data", exist_ok=True)
wfdb.dl_database('mitdb', 'mitbih_data')

print("Done! Dataset saved to 'mitbih_data' folder")
print("Next: python preprocess.py")