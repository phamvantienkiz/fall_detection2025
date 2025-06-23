import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf

# === Config ===
data_test_path = "../data_test"
CSV_FILE = os.path.join(data_test_path, "accel_20250518_092055.csv")  # Dữ liệu từ orange pi

model_paths = "../du_md_models/models"
SCALER_PATH = os.path.join(model_paths, "scaler_bi_v3v6.pkl")
TFLITE_MODEL_PATH = "best_model_bi_v3v6.tflite"
WINDOW_SIZE = 90
STRIDE = 10

# === Load và xử lý dữ liệu CSV ===
df = pd.read_csv(CSV_FILE)
# df = df[["X-Acc", "Y-Acc", "Z-Acc"]].dropna()
df = df[["x", "y", "z"]].dropna()
data = df.to_numpy()

print(f"[INFO] Dữ liệu đầu vào: {data.shape}")

# === Load scaler ===
scaler = joblib.load(SCALER_PATH)

# === Load TensorFlow Lite model ===
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Lấy thông tin input/output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Tạo sliding window và dự đoán ===
results = []

for start in range(0, len(data) - WINDOW_SIZE + 1, STRIDE):
    window = data[start:start + WINDOW_SIZE]
    window_scaled = scaler.transform(window)
    window_input = np.expand_dims(window_scaled, axis=0).astype(np.float32)  # (1, 90, 3)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], window_input)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])  # (1, 1)
    pred = output_data[0][0]
    label = 1 if pred > 0.5 else 0

    results.append({
        "start_index": start,
        "prob_fall": float(pred),
        "predict_label": label
    })

# === Bước 5: In kết quả ===
for r in results:
    print(f"[{r['start_index']:>4}]  Dự đoán: {'FALL' if r['predict_label']==1 else 'ADL'} | Xác suất fall: {r['prob_fall']:.4f}")
