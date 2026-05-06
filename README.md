# Bot Account Classification (Binary Machine Learning)
This open-source project builds a simple machine-learning model to classify accounts as **bots (1)** or **humans (0)** using profile and activity features.
The project is designed for **learning and teaching**:
- data cleaning
- preprocessing pipelines
- model training and evaluation
- generating prediction outputs
---
## Project Structure
BOT-ACCOUNT-CLASSIFICATION/
│
├── train.csv # Training data (includes outcome)
├── test.csv # Test data (no outcome)
│
├── notebooks/
│ └── bot_account_classification.ipynb
│
├── outputs/
│ └── submissions.csv # Generated predictions
│
├── requirements.txt
└── README.md

---
## Dataset Description
You are given two datasets:
- **train.csv**
  - Contains features and the target column `outcome`
  - `outcome = 1` → bot  
  - `outcome = 0` → human 
- **test.csv**
  - Contains the same features but **no outcome column**
### Feature Types
- **Numeric:** `friends_count`, `posts_count`
- **Categorical/Binary:** `photo_added`, `dob_given`, `profile_private`, etc.
- Some fields contain the string **"Unknown"**, which must be handled properly.
---
## Objective
Build a machine-learning model that predicts whether an account is a **bot** or a **human**.
---
## Tasks (Learning Goals)
Users are expected to:
1. Load `train.csv` and `test.csv` using Pandas  
2. Fix data quality issues:
   - Replace `"Unknown"` with missing values (`NaN`)
   - Convert numeric columns safely  
3. Handle outliers:
   - Clip `friends_count` between the 1st and 99th percentiles  
4. Build a preprocessing + model pipeline:
   - Numeric: median imputation  
   - Categorical: most-frequent imputation + One-Hot Encoding  
5. Train and validate the model:
   - 80/20 train-validation split (stratified)
   - Evaluate using **F1 score**  
6. Train on the full dataset and generate predictions for `test.csv`  
7. Create `submissions.csv` with exactly:
   - `id`
   - `outcome`
---
## Installation
Create a virtual environment (recommended), then install dependencies:
```bash
pip install -r requirements.txt
