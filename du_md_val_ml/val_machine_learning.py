import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.signal import welch
from pywt import wavedec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

DATASET_DIR = "du_md_dataset"
dataset = np.load(os.path.join(DATASET_DIR, "dataset_raw_w101.npy"))
labels = np.load(os.path.join(DATASET_DIR, "labels_raw_w101.npy"))

def extract_features(data):
    features = []
    for sample in data:  # sample: (101, 3)
        x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]
        feat = []

        # 1. Thống kê cơ bản
        for signal in [x, y, z]:
            feat.extend([
                np.mean(signal),  # Mean
                np.median(signal),  # Median
                np.std(signal),  # STD
                np.min(signal),  # Min
                np.max(signal),  # Max
                np.percentile(signal, 75) - np.percentile(signal, 25),  # IQR
                stats.entropy(np.histogram(signal, bins=10, density=True)[0])  # Entropy
            ])

        # 2. Wavelet
        for signal in [x, y, z]:
            coeffs = wavedec(signal, 'sym2', level=5)  # Symlet wavelet
            cA = coeffs[0]
            cD = np.concatenate(coeffs[1:])
            feat.append(np.mean(cA))  # MeanCA
            feat.append(np.mean(cD))  # MeanCV

        # 3. Vector Sum
        vs = np.sqrt(x**2 + y**2 + z**2)
        feat.extend([
            np.std(vs),  # VSSTD
            np.mean(vs),  # VSMean
            np.min(vs),  # VSMin
            np.max(vs),  # VSMax
            np.percentile(vs, 75) - np.percentile(vs, 25)  # VSIQR
        ])

        # 4. Slope Change và Zero Crossing
        for signal in [x, y, z]:
            slopes = np.diff(signal)
            slope_changes = np.sum(np.diff(np.sign(slopes)) != 0)
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            feat.extend([slope_changes, zero_crossings])

        # 5. Kurtosis và Skewness
        for signal in [x, y, z]:
            feat.append(stats.kurtosis(signal))  # Kurtosis
            feat.append(stats.skew(signal))  # Skewness

        # 6. Range
        for signal in [x, y, z]:
            feat.append(np.max(signal) - np.min(signal))  # Range

        # 7. PCA
        pca = PCA(n_components=3)
        pca.fit(sample)  # sample: (101, 3)
        feat.extend(pca.transform(sample).mean(axis=0))  # PCA scores cho 3 thành phần

        # 8. Dominant Sign
        for signal in [x, y, z]:
            ds = 1 if np.sum(signal >= 0) >= np.sum(signal < 0) else 0
            feat.append(ds)

        # 9. Normalization-based
        for signal in [x, y, z]:
            # MeanVecNorm
            vs_norm = signal / np.sqrt(np.sum(signal**2))
            feat.append(np.mean(vs_norm))
            # MeanZScore
            z_score = (signal - np.mean(signal)) / np.std(signal)
            feat.append(np.mean(z_score))
            # MeanRemove
            mean_removed = signal - np.mean(signal)
            feat.append(np.mean(mean_removed))
            # VecNormMeanRem
            vec_norm_mean_rem = vs_norm - np.mean(vs_norm)
            feat.append(np.mean(vec_norm_mean_rem))

        features.append(feat)
    return np.array(features)  # (N, 85)

features = extract_features(dataset)

# Lấy chỉ số của 39 đặc trưng được chọn (theo Bảng 5)
selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38]
features_39 = features[:, selected_indices]  # Chọn 39 đặc trưng theo Bảng 5

# Gộp nhãn
def merge_fall_labels(labels):
    new_labels = labels.copy()
    fall_classes = ['Falling Unconsciousness', 'Falling HeartAttack', 'Falling Slipping']
    adl_classes = ['Walking', 'Sitting', 'Sleeping', 'Jogging', 'Staircase Up',
                   'Staircase Down', 'Standing']
    for i, label in enumerate(new_labels):
        if label in fall_classes:
            new_labels[i] = 1
        elif label in adl_classes:
            new_labels[i] = 0
        else:
            print(f"Warning: Unrecognized label '{label}' at index {i}")
            new_labels[i] = -1  # Gán giá trị mặc định để phát hiện lỗi
    return new_labels.astype(int)

labels_merged = merge_fall_labels(labels)


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(features_39, labels_merged, test_size=0.2, random_state=42, stratify=labels_merged, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test, shuffle=True)

# Chuẩn hoá
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Định nghĩa các mô hình

# 1. SVM
svm_model = SVC(C=1.0, kernel='rbf', gamma=1.0 / features_39.shape[1], shrinking=True, tol=0.001)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=10, min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                                  random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 3. K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5, leaf_size=30, metric='euclidean')
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)


# Hàm tính các chỉ số từ confusion matrix
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, specificity, f1


# Tính chỉ số cho từng mô hình
svm_metrics = calculate_metrics(y_test, svm_pred)
rf_metrics = calculate_metrics(y_test, rf_pred)
knn_metrics = calculate_metrics(y_test, knn_pred)

# In kết quả
print("SVM Metrics:")
print(f"Accuracy: {svm_metrics[0]:.3f}")
print(f"Precision: {svm_metrics[1]:.3f}")
print(f"Recall (Sensitivity): {svm_metrics[2]:.3f}")
print(f"Specificity: {svm_metrics[3]:.3f}")
print(f"F1 Score: {svm_metrics[4]:.3f}")
print("\nRandom Forest Metrics:")
print(f"Accuracy: {rf_metrics[0]:.3f}")
print(f"Precision: {rf_metrics[1]:.3f}")
print(f"Recall (Sensitivity): {rf_metrics[2]:.3f}")
print(f"Specificity: {rf_metrics[3]:.3f}")
print(f"F1 Score: {rf_metrics[4]:.3f}")
print("\nK-NN Metrics:")
print(f"Accuracy: {knn_metrics[0]:.3f}")
print(f"Precision: {knn_metrics[1]:.3f}")
print(f"Recall (Sensitivity): {knn_metrics[2]:.3f}")
print(f"Specificity: {knn_metrics[3]:.3f}")
print(f"F1 Score: {knn_metrics[4]:.3f}")

