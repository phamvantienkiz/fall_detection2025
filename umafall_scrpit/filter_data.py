import os
import pandas as pd
from io import StringIO

activity_label = {
    "ClappingHands": 1, "HandsUp": 2, "MakingACall": 3, "OpeningDoor": 4,
    "Sitting": 5, "Walking": 6, "Bending": 7, "Hopping": 8,
    "Jogging":9, "LyingDown":10, "GoDownstairs": 11, "GoUpstairs": 12,
    "backwardFall": 13, "forwardFall": 14, "lateralFall": 15
}

def get_activity_name(filename: str) -> str:
    parts = filename.split('_')
    if "ADL" in parts:
        index = parts.index("ADL")
    elif "Fall" in parts:
        index = parts.index("Fall")
    else:
        return "Unknown"

    if index + 1 < len(parts):
        activity_part = parts[index+1]
        return activity_part
    else:
        return "Unknown"


def get_label(activity_name: str)-> str:
    return activity_label.get(activity_name, -1)

def process_csv_file(csv_path: str) -> pd.DataFrame:
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_line = lines[40].strip()
    header_line = header_line.lstrip('%').strip()

    columns = [col.strip() for col in header_line.split(',')]

    data_lines = lines[41:]
    data_str = ''.join(data_lines)

    df = pd.read_csv(StringIO(data_str), names=columns)

    df_filtered = df[
        (df["Sensor ID"] == 3) &
        (df["Sensor Type"] == 0)
        ].copy()

    filename = os.path.basename(csv_path)
    activity_name = get_activity_name(filename)
    label_value = get_label(activity_name)
    df_filtered["label"] = label_value

    return df_filtered

def main():
    base_dir = "UMAFallDataset"

    adl_dir = os.path.join(base_dir, "ADL")
    fall_dir = os.path.join(base_dir, "Fall")

    adl_dfs = []
    fall_dfs = []

    if os.path.exists(adl_dir):
        for fname in os.listdir(adl_dir):
            if fname.lower().endswith(".csv"):
                csv_path = os.path.join(adl_dir, fname)
                df_adl = process_csv_file(csv_path)
                if not df_adl.empty:
                    adl_dfs.append(df_adl)

    if os.path.exists(fall_dir):
        for fname in os.listdir(fall_dir):
            if fname.lower().endswith(".csv"):
                csv_path = os.path.join(fall_dir, fname)
                df_fall = process_csv_file(csv_path)
                if not df_fall.empty:
                    fall_dfs.append(df_fall)

    features = ['X-Axis', 'Y-Axis', 'Z-Axis']
    if adl_dfs:
        df_all_adl = pd.concat(adl_dfs, ignore_index=True)
        df_all_adl[features] = df_all_adl[features].astype('float64')
        for col in features:
            print(f"\nCột '{col}' ADL:")
            print(f"  - Kiểu dữ liệu: {df_all_adl[col].dtype}")
        # df_all_adl.to_csv("ADL_Wrist_Acc.csv", index=False)
        # print("Da luu ADL_Wrist_acc.csv")

    if fall_dfs:
        df_all_fall = pd.concat(fall_dfs, ignore_index=True)
        for col in features:
            print(f"\nCột '{col}' Fall:")
            print(f"  - Kiểu dữ liệu: {df_all_fall[col].dtype}")
        # df_all_fall.to_csv("FALL_Wrist_Acc.csv", index=False)
        # print("Da luu FALL_Wrist_Acc.csv")

if __name__ == "__main__":
    main()
