import pandas as pd
import numpy as np
from collections import Counter


def check_data_quality(df, columns):
    """Kiểm tra NaN và kiểu dữ liệu"""
    print("=== KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ===")

    for col in columns:
        print(f"\nCột '{col}':")
        print(f"  - Kiểu dữ liệu: {df[col].dtype}")
        print(f"  - Số lượng NaN: {df[col].isna().sum()}")
        print(f"  - Tổng số mẫu: {len(df[col])}")

        if df[col].isna().sum() > 0:
            print(f"  - Tỷ lệ NaN: {df[col].isna().sum() / len(df[col]) * 100:.2f}%")


def analyze_sequences_by_sample_no(df):
    """Phân tích chuỗi dữ liệu dựa trên Sample No"""
    print("\n=== PHÂN TÍCH CHUỖI DỰA TRÊN SAMPLE NO ===")

    sequences = []
    current_sequence_start = 0

    for i in range(1, len(df)):
        # Khi Sample No reset về 0, nghĩa là bắt đầu chuỗi mới
        if df['Sample No'].iloc[i] == 0:
            # Lưu thông tin chuỗi trước đó
            sequence_length = i - current_sequence_start
            sequence_label = df['label'].iloc[current_sequence_start]
            max_sample_no = df['Sample No'].iloc[i - 1]

            sequences.append({
                'label': sequence_label,
                'start_idx': current_sequence_start,
                'end_idx': i,
                'length': sequence_length,
                'max_sample_no': max_sample_no
            })

            current_sequence_start = i

    # Thêm chuỗi cuối cùng
    sequence_length = len(df) - current_sequence_start
    sequence_label = df['label'].iloc[current_sequence_start]
    max_sample_no = df['Sample No'].iloc[-1]

    sequences.append({
        'label': sequence_label,
        'start_idx': current_sequence_start,
        'end_idx': len(df),
        'length': sequence_length,
        'max_sample_no': max_sample_no
    })

    print(f"Tổng số chuỗi tìm được: {len(sequences)}")

    # Thống kê
    lengths = [seq['length'] for seq in sequences]
    max_sample_nos = [seq['max_sample_no'] for seq in sequences]

    print(f"Độ dài chuỗi - Min: {min(lengths)}, Max: {max(lengths)}, Trung bình: {np.mean(lengths):.2f}")
    print(
        f"Sample No cuối - Min: {min(max_sample_nos)}, Max: {max(max_sample_nos)}, Trung bình: {np.mean(max_sample_nos):.2f}")

    # Hiển thị một số chuỗi đầu tiên để kiểm tra
    print("\nMột số chuỗi đầu tiên:")
    for i, seq in enumerate(sequences[:5]):
        print(f"  Chuỗi {i + 1}: Label {seq['label']}, Length {seq['length']}, Max Sample No: {seq['max_sample_no']}")

    return sequences


def create_windows(df, window_size, features, label_col):
    """
    Tạo windows cho LSTM từ dữ liệu time series dựa trên Sample No

    Args:
        df: DataFrame chứa dữ liệu
        window_size: Kích thước cửa sổ
        features: Danh sách tên cột features
        label_col: Tên cột label

    Returns:
        windows: np.array shape (n_windows, window_size, n_features)
        labels: np.array shape (n_windows,)
    """
    print(f"\n=== TẠO WINDOWS (window_size={window_size}) ===")

    df[features] = df[features].astype('float32')

    # Phân tích chuỗi dựa trên Sample No
    sequences = analyze_sequences_by_sample_no(df)

    windows = []
    window_labels = []

    processed_sequences = 0
    skipped_sequences = 0
    total_windows = 0

    for seq in sequences:
        seq_label = seq['label']
        seq_length = seq['length']
        seq_start = seq['start_idx']
        seq_end = seq['end_idx']
        max_sample_no = seq['max_sample_no']

        print(f"\nChuỗi label {seq_label}: {seq_length} mẫu (Sample No 0-{max_sample_no}):")

        if seq_length >= 270:  # Đủ dữ liệu để tạo ít nhất 3 windows (270/90=3)
            # Tính số windows có thể tạo
            n_windows = seq_length // window_size
            usable_samples = n_windows * window_size
            remaining_samples = seq_length - usable_samples

            print(f"  - Có thể tạo {n_windows} windows")
            print(f"  - Sử dụng {usable_samples} mẫu, bỏ qua {remaining_samples} mẫu cuối")

            # Tạo windows từ chuỗi này
            for i in range(n_windows):
                window_start = seq_start + i * window_size
                window_end = window_start + window_size

                # Đảm bảo không vượt quá ranh giới của chuỗi
                if window_end <= seq_end:
                    # Lấy dữ liệu features cho window
                    window_data = df[features].iloc[window_start:window_end].values

                    # Kiểm tra xem window có đủ dữ liệu không
                    if len(window_data) == window_size:
                        windows.append(window_data)
                        window_labels.append(seq_label)
                        total_windows += 1

            processed_sequences += 1

        else:
            print(f"  - Bỏ qua (chỉ có {seq_length} mẫu, cần ít nhất 270 mẫu)")
            skipped_sequences += 1

    print(f"\nTóm tắt:")
    print(f"  - Đã xử lý: {processed_sequences} chuỗi")
    print(f"  - Đã bỏ qua: {skipped_sequences} chuỗi")
    print(f"  - Tổng số windows tạo được: {total_windows}")

    # Chuyển đổi thành numpy arrays
    windows = np.array(windows)
    window_labels = np.array(window_labels)

    print(f"  - Shape của windows: {windows.shape}")
    print(f"  - Shape của labels: {window_labels.shape}")

    return windows, window_labels


def main():
    # Đọc file CSV
    print("Đọc file CSV...")
    df_adl = pd.read_csv("ADL_Wrist_Acc.csv")
    df_fall = pd.read_csv("FALL_Wrist_Acc.csv")

    # Kết hợp dữ liệu
    df = pd.concat([df_adl, df_fall], ignore_index=True)
    print(f"Tổng số mẫu: {len(df)}")

    # Định nghĩa features và labels
    features = ['X-Axis', 'Y-Axis', 'Z-Axis']
    label_col = 'label'
    window_size = 60

    print(f"Features: {features}")
    print(f"Label column: {label_col}")
    print(f"Window size: {window_size}")

    # Kiểm tra chất lượng dữ liệu
    columns_to_check = features + [label_col]
    check_data_quality(df, columns_to_check)

    # Kiểm tra xem có cột Sample No không
    if 'Sample No' not in df.columns:
        print("CẢNH BÁO: Không tìm thấy cột 'Sample No'. Không thể xác định ranh giới chuỗi chính xác.")
        return
    else:
        print(f"Tìm thấy cột 'Sample No' với {df['Sample No'].nunique()} giá trị unique")

    # Xử lý NaN nếu có
    nan_count = df[columns_to_check].isna().sum().sum()
    if nan_count > 0:
        print(f"\nCảnh báo: Có {nan_count} giá trị NaN. Đang xóa các dòng chứa NaN...")
        df = df.dropna(subset=columns_to_check)
        print(f"Số mẫu sau khi xóa NaN: {len(df)}")

    # Sắp xếp dữ liệu theo TimeStamp để đảm bảo thứ tự thời gian
    # if 'TimeStamp' in df.columns:
    #     df = df.sort_values('TimeStamp').reset_index(drop=True)
    #     print("Đã sắp xếp dữ liệu theo TimeStamp")

    # Tạo windows
    windows, labels = create_windows(df, window_size, features, label_col)

    # Lưu kết quả
    print("\nLưu kết quả...")
    # np.save("data_umaf_w60.npy", windows)
    # np.save("labels_umaf_w60.npy", labels)

    print("Đã lưu thành công:")
    print(f"  - data_umaf_w60.npy: {windows.shape}")
    print(f"  - labels_umaf_w60.npy: {labels.shape}")

    # Thống kê cuối cùng
    print(f"\nThống kê label sau khi tạo windows:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} windows")


if __name__ == "__main__":
    main()