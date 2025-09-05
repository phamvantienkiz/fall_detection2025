# Fall Detection (2025)

This project is part of a graduate thesis focusing on **Fall Detection** using deep learning models and wrist-worn sensor datasets. Current scope includes data pre-processing, model training, evaluation and quantization.

## 📁 Project Structure

```
DATN_FD_2025/
├── data_test/
├── du_md_dataset/             # Preprocessed DU-MD dataset (.npy format)
├── du_md_models/              # Trained DU-MD models
├── du_md_script/              # Scripts for training DU-MD models
├── du_md_val_ml/
├── umafall_dataset/           # Preprocessed UMAFall dataset (.npy format)
├── umafall_models/            # Trained UMAFall models
├── umafall_script/            # Scripts for training UMAFall models
├── umafall_val_ml/
├── Mobile/fall_tracking       # Mobile app
├── model_quantization/        # Quantization scripts or results
├── report/                    # Report file
├── Segmented_Raw_Data/        # Raw sensor dataset (after download)
├── check_lib.py
├── test_models.py             # Model evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation

```

## 🚀 Getting Started

### 1\. Clone the repository or download the ZIP:

```
git clone https://github.com/phamvantienkiz/fall_detection2025.git
cd fall-detection-2025

```

### 2\. Create virtual environment:

```
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n fall_env python=3.10
conda activate fall_env

```

### 3\. Install dependencies:

```
pip install -r requirements.txt

```

### 4\. Download DU-MD Dataset

- Download the DU-MD dataset from the official website:\
  👉 <https://ahadvisionlab.com/mobility.html>

- Extract and place it into:

```
Segmented_Raw_Data/

```

### 5\. Preprocessed Data

- Preprocessed `.npy` data is already available in:

  - `du_md_dataset/`

  - `umafall_dataset/`

## 🧠 Train Models

### DU-MD Dataset:

1.  Navigate to:

```
cd du_md_script/

```

1.  Run training script:

```
python training.py

```

1.  Note:

    - Make sure to adjust file names and paths in `load_data.py` and `training.py`, including:

      - `scaler.pkl`

      - `best_model.h5`

      - `final_model.h5`

      - `confusion_matrix.png`

      - `history.png`

### UMAFall Dataset:

1.  Navigate to:

```
cd umafall_script/

```

1.  Run training script and adjust configs as above.

## 📊 Testing & Evaluation

- Use `test_models.py` for testing.

## ⚠️ Notes

- Ensure consistent file names and paths between scripts.

- Quantization logic (if needed) is available in `model_quantization/`

## Mobile Flutter Application

The `Mobile` directory contains the source code for the mobile application developed using Flutter, which is designed for fall detection and event tracking. This app can connect to the fall detection system, display results, and assist users in monitoring their health.

## Thesis Report

The `report` directory contains the thesis report file for the fall detection topic. The report presents detailed information about the research process, model development, experimental results, system architecture, and related applications.

## 📄 License

This project is part of an academic thesis and is for educational use only.

---

Feel free to modify and expand based on your model architecture, visualization tools, or backend integration!
