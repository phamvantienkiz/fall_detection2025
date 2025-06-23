import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# === Config ===
data_test_path = "data_test"
CSV_FILE = os.path.join(data_test_path, "accel_20250518_092105.csv")  # Dữ liệu từ orange pi

model_paths = "du_md_models/models"
SCALER_PATH = os.path.join(model_paths, "scaler_bi_v3v3.pkl")
MODEL_PATH = os.path.join(model_paths, "best_model_bi_v3v3.h5")
WINDOW_SIZE = 90
STRIDE = 10

# === Bước 1: Load và xử lý dữ liệu CSV ===
df = pd.read_csv(CSV_FILE)

# Chỉ giữ lại 3 cột accelerometer
# df = df[["X-Acc", "Y-Acc", "Z-Acc"]].dropna()
df = df[["x", "y", "z"]].dropna()
data = df.to_numpy()  # (num_samples, 3)

print(f"[INFO] Dữ liệu đầu vào: {data.shape}")

# === Bước 2: Load mô hình và scaler ===
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)

# === Bước 3: Tạo sliding window ===
results = []

for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
    window = data[start:start + WINDOW_SIZE]  # (90, 3)

    # Chuẩn hóa
    window_scaled = scaler.transform(window)  # (90, 3)

    # Định dạng cho model: (1, 90, 3)
    window_input = np.expand_dims(window_scaled, axis=0)

    # Dự đoán
    pred = model.predict(window_input)[0][0]  # sigmoid → 1 giá trị xác suất
    label = 1 if pred > 0.5 else 0

    results.append({
        "start_index": start,
        "prob_fall": float(pred),
        "predict_label": label
    })

# === Bước 4: In kết quả ===
for r in results:
    print(f"[{r['start_index']:>4}]  Dự đoán: {'FALL' if r['predict_label']==1 else 'ADL'} | Xác suất fall: {r['prob_fall']:.4f}")
