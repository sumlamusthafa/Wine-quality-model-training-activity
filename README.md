# Red Wine Quality Prediction

Machine learning activity (Tutorial 9 – Model Training). Predicts whether a red wine is
**good (quality ≥ 7)** or **bad** from its chemical properties.

## Dataset
Red Wine Quality dataset (UCI Machine Learning Repository) – 1,599 wines, 11 chemical features.
Only 217 wines (~14%) are "good", so the classes are heavily imbalanced.

## Approach
1. Exploratory data analysis (missing values, correlation heatmap)
2. Converted quality to binary: good (≥ 7) / bad
3. Stratified 80/20 train-test split
4. SMOTE oversampling and standard scaling applied **on the training set only**
   (to avoid data leakage)
5. Compared 6 models on 3 feature sets (all 11, top 6 by correlation, reduced set)
6. Hyperparameter tuning with GridSearchCV (5-fold CV)

## Results

| Model | Accuracy | F1 (good wine) |
|---|---|---|
| **Random Forest** | **0.916** | **0.697** |
| Gradient Boosting | 0.878 | 0.606 |
| SVM (RBF) | 0.866 | 0.598 |
| Decision Tree | 0.850 | 0.556 |
| Logistic Regression | 0.825 | 0.548 |
| KNN | 0.816 | 0.543 |

**Best model:** Random Forest (n_estimators=400, max_depth=None, min_samples_leaf=1)
with all 11 features. Most important features: alcohol, sulphates, volatile acidity.

Accuracy alone can mislead here (predicting "bad" for every wine gives ~87%),
so F1-score on the good-wine class is also reported.

**Note on data leakage:** The tutorial applies SMOTE before the train-test split, which gives
a higher accuracy (~93%) because synthetic samples derived from training points can leak into
the test set. Here SMOTE is applied only to the training data, so the 91.6% reported is a more
realistic estimate of performance on unseen wines.

## Files
- `Tutorial9_Notebook.ipynb` – full code and outputs (rename to match your uploaded file)
- `winequality-red.csv` – dataset

## Run it
```bash
pip install pandas scikit-learn imbalanced-learn seaborn matplotlib
jupyter notebook
```
