# Predictive Maintenance using Random Forest

This project focuses on building a machine learning model to predict equipment failure in advance using sensor data. The model uses a Random Forest classifier trained on NASA's CMAPSS dataset, commonly used for predictive maintenance research.

## 📌 Objective

To predict whether a machine (e.g., jet engine unit) is likely to fail within the next 30 operational cycles. This helps in performing timely maintenance and avoiding costly failures.

## 📊 Dataset

We use the **CMAPSS** dataset, specifically:
- `PM_train.txt` – Training data with operational cycles and sensor readings.
- `PM_test.txt` – Test data without failure labels.
- `truth.txt` – Remaining useful life (RUL) values for test units.

Each row in the dataset represents a single time-cycle reading for a machine.

### Features
- 3 Operational Settings (`setting1`, `setting2`, `setting3`)
- 21 Sensor Readings (`s1` to `s21`)
- Target: Binary label (`label_bc`) indicating whether a failure is expected within 30 cycles.

## 🧠 Machine Learning Model

We use a **Random Forest Classifier** for binary classification:

### Why Random Forest?
- Handles tabular sensor data well
- Less sensitive to hyperparameters
- Provides feature importance
- Robust and interpretable

### Steps:
1. **Data Preprocessing**
   - Merging test set with truth data to calculate `ttf` (time-to-failure)
   - Creating binary labels: 1 if `ttf ≤ 30`, else 0
   - Feature scaling with `MinMaxScaler`

2. **Training**
   - Model trained on historical sequences from training units
   - Used `RandomForestClassifier` from `scikit-learn`
   - Performed hyperparameter tuning using `GridSearchCV`

3. **Evaluation**
   - Accuracy: ~98%
   - Cross-validated Accuracy: 96% ± 1%
   - Confusion matrix and classification report used for performance evaluation

## 📈 Results

- **High prediction accuracy** using Random Forest
- **Feature importance** reveals which sensors contribute most to failure prediction

## 🔍 Key Files

| File | Description |
|------|-------------|
| `PM_train.txt` | Training dataset with failure history |
| `PM_test.txt` | Test dataset without labels |
| `truth.txt` | True remaining cycles before failure for test units |
| `predictive_maintenance_rf.ipynb` | Main notebook with model training and evaluation |
| `README.md` | Project documentation |

## 🛠️ How to Run

1. Clone the repo and install requirements (`scikit-learn`, `pandas`, `numpy`, etc.)
2. Run the notebook step-by-step or execute all cells
3. Use the `prob_failure(machine_id)` function to predict failure probability of any machine

## 📌 Sample Output

```bash
Probability that machine 4 will fail within 30 days: 87.5%
