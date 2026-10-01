from pathlib import Path
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI(title="Wi-Fi CSI Obfuscator API")

# Ensure paths resolve relative to this script's directory
BASE_DIR = Path(__file__).resolve().parent

# Load Production Artifacts safely
rf_model = joblib.load(BASE_DIR / "blackbox_rf_model.joblib")
uap_hardware = np.load(BASE_DIR / "quantized_uap.npy")
test_data = np.load(BASE_DIR / "real_test_waves.npz")
real_waves = test_data['waves']
real_labels = test_data['labels']

@app.get("/intercept")
def intercept_wave(wave_id: int = 0):
    """Fetches a real physical RF wave from the test environment"""
    try:
        wave_idx = wave_id % len(real_waves)
        clean_wave = real_waves[wave_idx]
        true_label = int(real_labels[wave_idx])
        
        # Flatten for Random Forest ingestion
        flat_wave = clean_wave.reshape(1, -1)
        
        # 1. Baseline Eavesdropper Prediction
        clean_guess = int(rf_model.predict(flat_wave)[0])
        clean_probs = rf_model.predict_proba(flat_wave)[0]
        clean_conf = float(max(clean_probs) * 100)

        # 2. Apply the Hardware-Quantized UAP Shield
        obf_wave = clean_wave + uap_hardware[0]
        flat_obf = obf_wave.reshape(1, -1)
        
        # 3. Defeated Eavesdropper Prediction
        obf_guess = int(rf_model.predict(flat_obf)[0])
        obf_probs = rf_model.predict_proba(flat_obf)[0]
        obf_conf = float(max(obf_probs) * 100)

        return {
            "true_label": true_label,
            "clean_guess": clean_guess,
            "clean_confidence": clean_conf,
            "obf_guess": obf_guess,
            "obf_confidence": obf_conf,
            "original_wave": clean_wave.tolist(),
            "scrambled_wave": obf_wave.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))