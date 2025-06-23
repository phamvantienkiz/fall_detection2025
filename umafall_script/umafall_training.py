from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from umafall_bi_model import build_model
from umafall_load_data import load_dataset

num_classes = 1

def train_model(data_dir, seq_length=90, num_classes=num_classes, epochs=20, batch_size=32):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_dataset(
        data_dir=data_dir,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        apply_augment=True,
        noise_std=0.01,
        num_augments=1
    )


    print("[DEBUG] Checking for NaNs or Infs in X_train:")
    print("NaNs:", np.isnan(X_train).sum(), "Infs:", np.isinf(X_train).sum())
    print("Mean:", np.mean(X_train), "Std:", np.std(X_train))

    print("Unique labels:", np.unique(y_train))
    print("Labels dtype:", y_train.dtype)
    print("Max label:", np.max(y_train), "Expected:", num_classes)

    # Tính toán class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class Weights:", class_weight_dict)

    # Lấy input_shape từ dữ liệu
    input_shape = (seq_length, X_train.shape[2])  # (timesteps, features)

    # Xây dựng mô hình CB-LSTM
    model = build_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    # Đường dẫn lưu mô hình
    save_dir = "../umafall_models"
    model_dir = os.path.join(save_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Lưu mô hình tốt nhất
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, "best_model_bi_umaf_v4v9.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weight_dict
    )

    # Lưu mô hình cuối cùng sau huấn luyện
    final_model_path = os.path.join(model_dir, "final_model_bi_umaf_v4v9.h5")
    model.save(final_model_path)

    # Vẽ biểu đồ Accuracy và Loss theo từng epoch
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    save_result = "../umafall_models/results"
    plt.savefig(os.path.join(save_result, "history_model_bi_umaf_v4v9.png"))
    plt.show()

    # Đánh giá mô hình trên tập test
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nĐộ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

    # Dự đoán trên tập test
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Báo cáo classification
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ADL", "Fall"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    print(f"\n✅ Sensitivity (Recall cho Fall): {sensitivity:.2f}")
    print(f"✅ Specificity (Recall cho ADL): {specificity:.2f}")

    # Tính ROC và AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    print(f"✅ AUC: {roc_auc:.2f}")

    # Ve CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ADL", "Fall"], yticklabels=["ADL", "Fall"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    save_result = "../umafall_models/results"
    plt.savefig(os.path.join(save_result, "cm_model_bi_umaf_v4v9.png"))
    plt.show()

    # Vẽ ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_result, "roc_model_bi_umaf_v4v9.png"))
    plt.show()


if __name__ == "__main__":
    data_dir = "../umafall_dataset"
    seq_length = 90
    epochs = 50
    batch_size = 32
    train_model(data_dir=data_dir, seq_length=seq_length, num_classes=num_classes, epochs=epochs, batch_size=batch_size)
