# Assignment 4: Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
To develop a K-Nearest Neighbors (KNN) classification model that predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements, for a healthcare organization use case.

## Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset**
Kaggle link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

The notebook loads the dataset via `sklearn.datasets.load_breast_cancer()`, which contains the exact same 569 records and 30 diagnostic features (radius, texture, perimeter, area, smoothness, concavity, etc., computed as mean / standard error / "worst" values) as the Kaggle CSV, sourced from the same UCI ML Repository entry. A commented-out alternate loading cell (`pd.read_csv("data.csv")`) is included in the notebook if you prefer to work directly from the downloaded Kaggle file — the rest of the pipeline is unchanged either way. The raw dataset file is **not** included in this repository; please download it from the Kaggle link above if needed.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — visualization (class distribution, confusion matrix, K-vs-accuracy plot)
- `scikit-learn` — dataset, preprocessing (`StandardScaler`, `LabelEncoder`), model (`KNeighborsClassifier`), `train_test_split`, evaluation metrics

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, identified 30 numerical features and the categorical target (`diagnosis`: M/B), and reviewed dataset info and summary statistics.
2. **Data Preprocessing**
   - Checked for missing values (none found).
   - Dropped non-feature columns (`id`, `Unnamed: 32` when using the raw Kaggle CSV).
   - Encoded the target variable using `LabelEncoder` (B → 0, M → 1).
   - Standardized all feature values using `StandardScaler` (critical for a distance-based algorithm like KNN).
   - Split the data into 80% training and 20% testing sets, stratified by class.
3. **Model Development** — Trained a `KNeighborsClassifier` with **K = 5** on the standardized training data and generated predictions on the test set.
4. **Model Evaluation** — Computed Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix. Also explored accuracy across K = 1 to 20 to understand the effect of K on performance.

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.9561 |
| Precision | 0.9744 |
| Recall | 0.9048 |
| F1-Score | 0.9383 |

**Confusion Matrix (K=5):**

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 71 | 1 |
| **Actual Malignant** | 4 | 38 |

**Observations:**
1. The KNN model (K=5) achieves ~95.6% accuracy on the test set, showing the diagnostic features are strongly separable between Malignant and Benign classes after standardization.
2. Precision (0.97) and Recall (0.90) are both high, meaning the model rarely misclassifies benign tumors as malignant, and correctly catches most malignant tumors; only 5 of 114 test samples were misclassified.
3. Testing K from 1–20 showed accuracy is fairly stable across K = 5–11, while very small K (K=1) can overfit to noise and very large K can underfit and blur the decision boundary.

## Conclusion
The KNN classifier built on the Breast Cancer Wisconsin (Diagnostic) dataset performed strongly at distinguishing malignant from benign tumors, achieving high accuracy, precision, recall, and F1-score with K = 5. The confusion matrix confirmed that misclassifications were rare, meaning the diagnostic measurements (radius, texture, perimeter, area, concavity, etc.) carry strong predictive signal for tumor classification.

Feature scaling was essential for this model. KNN classifies a point based on the Euclidean distance to its nearest neighbors, so features measured on larger numeric scales (e.g., area) would otherwise dominate the distance calculation and drown out equally important but smaller-scale features (e.g., smoothness). Standardizing all features to a common scale ensured every feature contributed fairly to the distance metric, directly improving classification quality.

One key limitation of KNN is its computational cost at prediction time: since it must calculate the distance from a new point to every training sample, it scales poorly with very large datasets and high-dimensional feature spaces, and it also requires careful tuning of K to balance overfitting and underfitting.

## Repository Contents
- `Assignment-4.ipynb` — full notebook with code, outputs, and visualizations
- `README.md` — this file
