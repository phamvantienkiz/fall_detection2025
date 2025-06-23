import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def augment_fall_data(X, y, noise_std=0.01, num_augments=1):
    fall_indices = np.where(y == 1)[0]
    X_fall = X[fall_indices]
    y_fall = y[fall_indices]

    X_augmented = []
    y_augmented = []

    for _ in range(num_augments):
        noise = np.random.normal(loc=0, scale=noise_std, size=X_fall.shape)
        X_noisy = X_fall + noise
        X_augmented.append(X_noisy)
        y_augmented.append(y_fall.copy())

    X_augmented = np.concatenate(X_augmented, axis=0)
    y_augmented = np.concatenate(y_augmented, axis=0)

    return X_augmented, y_augmented

def load_dataset(data_dir="../du_md_dataset", test_size=0.1, val_size=0.1, random_state=42,
                 apply_augment=True, noise_std=0.01, num_augments=2):
    # Load dữ liệu
    X = np.load(os.path.join(data_dir, "dataset_3400_ws90_s10.npy"))  # (samples, 90, 3)
    y = np.load(os.path.join(data_dir, "labels_3400_ws90_s10.npy"))

    # Gộp nhãn thành ADL = 0, Fall = 1
    merge_map = {
        "Walking": 0, "Sitting": 0, "Sleeping": 0, "Jogging": 0, "Staircase Up": 0, "Staircase Down": 0, "Standing": 0,
        # ADL
        "Falling Unconsciousness": 1, "Falling HeartAttack": 1, "Falling Slipping": 1  # FALL
    }

    # Apply mapping
    y = np.vectorize(merge_map.get)(y)
    print(f"[INFO] Các nhãn sau khi gộp: {np.unique(y)}")

    print(f"[INFO] Dữ liệu gốc: X shape = {X.shape}, y shape = {y.shape}")

    # Chia train + temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size,
        random_state=random_state,
        stratify=y,
        shuffle=True
    )

    # Chia temp → val và test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        random_state=random_state,
        stratify=y_temp,
        shuffle=True
    )

    # Chuẩn hóa dựa trên tập train
    num_samples, seq_len, num_features = X_train.shape

    # Reshape để scale theo feature
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_val_reshaped = X_val.reshape(-1, num_features)
    X_test_reshaped = X_test.reshape(-1, num_features)

    # Chuẩn hóa với scaler từ tập train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape lại về (samples, 90, 3)
    X_train = X_train_scaled.reshape(X_train.shape)
    X_val = X_val_scaled.reshape(X_val.shape)
    X_test = X_test_scaled.reshape(X_test.shape)

    # Lưu scaler
    joblib.dump(scaler, '../du_md_models/models/scaler_bi_v3v7.pkl')

    # Augment dữ liệu FALL nếu bật
    if apply_augment:
        X_aug, y_aug = augment_fall_data(X_train, y_train, noise_std=noise_std, num_augments=num_augments)
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)
        print(f"[INFO] Sau augment: X_train = {X_train.shape}, y_train = {y_train.shape}")

    print(f"[INFO] Đã chia và chuẩn hóa:")
    print(f" - Train: {X_train.shape}, {y_train.shape}")
    print(f" - Val:   {X_val.shape}, {y_val.shape}")
    print(f" - Test:  {X_test.shape}, {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
