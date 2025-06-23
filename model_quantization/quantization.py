import tensorflow as tf
from tensorflow.keras.models import load_model

# Tải mô hình Keras
model = load_model('../du_md_models/models/best_model_bi_v3v6.h5')

# Chuyển đổi sang TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.OpsSet.TFLITE_BUILTINS]  # Tối ưu hóa kích thước mô hình
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

# Lưu mô hình TensorFlow Lite
with open('best_model_bi_v3v6.tflite', 'wb') as f:
    f.write(tflite_model)

print("Mô hình đã được chuyển đổi và lưu thành công.")
