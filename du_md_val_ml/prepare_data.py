import pandas as pd
import numpy as np
import os
import re


# Đọc dữ liệu từ một tệp .TXT
def read_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    return data.values  # Ma trận 101x3 (X, Y, Z)


# Đọc toàn bộ dữ liệu từ thư mục
def load_dataset(data_dir):
    dataset = []
    labels = []
    for subject_folder in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_folder)
        for activity_folder in os.listdir(subject_path):
            activity_path = os.path.join(subject_path, activity_folder)
            activity_name = activity_folder.split("_", 1)[-1]
            for file_name in os.listdir(activity_path):
                file_path = os.path.join(activity_path, file_name)
                #print(f"{file_path} Done!")
                data = read_data(file_path)  # Ma trận 101x3
                if data.shape == (101, 3):
                    dataset.append(data)
                    labels.append(activity_name)  # Nhãn là tên thư mục con (ADL/fall)
                else:
                    print(f"=> {file_path} shape # (101, 3) line!!!")
    return np.array(dataset), np.array(labels)  # dataset: (N, 90, 3), labels: (N,)

data_dir = "../Segmented_Raw_Data"
dataset, labels = load_dataset(data_dir)

# Save dataset
OUTPUT_DIR = "../du_md_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "dataset_raw_w101.npy"), dataset)
np.save(os.path.join(OUTPUT_DIR, "labels_raw_w101.npy"), labels)

print("Đã xử lý xong dữ liệu DU-MD!")
print(f"- Tổng số mẫu: {len(dataset)}")
print(f"- Số nhãn: {len(set(labels))}")
print(f"- Đã lưu tại: {OUTPUT_DIR}")

print(dataset.shape)
#--------------------------------------------------

