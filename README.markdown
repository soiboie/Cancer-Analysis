# Cancer Analysis Project

## Overview
This project focuses on analyzing cancer-related data by preprocessing datasets, handling missing values, and applying machine learning models to predict cancer diagnosis. The dataset is sourced from multiple Excel files, combined, and cleaned for analysis. Various machine learning algorithms, including K-Nearest Neighbors (KNN) Classifier, KNN Regressor, Random Forest Classifier, and Logistic Regression, are implemented to evaluate their performance on the dataset.

## Project Structure
- **Cancer_Analysis.ipynb**: The main Jupyter Notebook containing the data preprocessing, cleaning, and machine learning analysis.
- **Cancer data.xlsx**, **Cancer 2 data.xlsx**: Input datasets containing cancer-related features (e.g., Age, Gender, Smoking, Genetic Risk, Alcohol Intake, BMI, Physical Activity, Cancer History, Diagnosis).
- **Meta Cancer data.xlsx**: The output dataset after preprocessing and imputing missing values.

## Dependencies
To run the project, ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imblearn`
- `openpyxl` (for Excel file handling)

You can install the dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imblearn openpyxl
```

## Data Preprocessing
1. **Data Loading**: 
   - Excel files (`Cancer data.xlsx`, `Cancer 2 data.xlsx`) are loaded from a specified directory.
   - The datasets are combined into a single DataFrame (`combined_cancer_df`).

2. **Data Standardization**:
   - **Gender**: Converted to 'M' (Male) or 'F' (Female) based on file-specific mappings.
   - **Smoking**: Standardized to 'Yes' or 'No' using mappings tailored to each dataset.
   - **Genetic Risk**: Converted to 'Yes' or 'No' based on numerical values.
   - **Alcohol Intake**: Transformed into 'Yes' or 'No' based on threshold values.

3. **Handling Missing Values**:
   - Categorical variables (`Gender`, `Smoking`, `Genetic Risk`, `Alcohol Intake`, `CancerHistory`, `Diagnosis`) are encoded using `LabelEncoder`.
   - Missing values are imputed using the KNN Imputer with 5 neighbors.
   - Encoded categorical columns are decoded back to their original values post-imputation.

4. **Data Export**:
   - The preprocessed dataset is saved as `Meta Cancer data.xlsx`.

## Machine Learning Analysis
The project evaluates multiple machine learning models for cancer diagnosis prediction using the Iris dataset as a placeholder (Note: The notebook uses the Iris dataset for model testing, which should be replaced with the actual cancer dataset for real analysis). The models include:

1. **K-Nearest Neighbors (KNN) Classifier**:
   - Trained with `n_neighbors=5`.
   - Achieved 100% accuracy on the test set.

2. **K-Nearest Neighbors (KNN) Regressor**:
   - Trained with `n_neighbors=5`.
   - Achieved 98.86% accuracy on the test set.

3. **Random Forest Classifier**:
   - Trained with default parameters.
   - Achieved 100% accuracy on the test set.

4. **Logistic Regression**:
   - Trained on SMOTE-resampled data to handle class imbalance.
   - Achieved 100% accuracy with perfect precision, recall, and F1-scores.

**Note**: The notebook currently uses the Iris dataset (`load_iris`) for model training and evaluation. To perform actual cancer analysis, replace `X` and `y` with the preprocessed cancer dataset (`df_imputed`) and ensure the target variable (`Diagnosis`) is used.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cancer-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd cancer-analysis
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the input Excel files (`Cancer data.xlsx`, `Cancer 2 data.xlsx`) in the appropriate directory (e.g., `/content/drive/MyDrive/Cancer Data`).
5. Open and run the `Cancer_Analysis.ipynb` notebook in Jupyter or Google Colab.
6. The preprocessed dataset will be saved as `Meta Cancer data.xlsx`.

## Issues and Notes
- The notebook includes Google Colab-specific code (e.g., `files.download`). Comment out these lines if running locally.
- The Logistic Regression model raises a convergence warning, suggesting an increase in `max_iter` or data scaling. Consider adding:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  ```
- Replace the Iris dataset with the actual cancer dataset (`df_imputed`) for meaningful analysis.

## Future Improvements
- Replace the Iris dataset with the actual cancer dataset for model training and evaluation.
- Perform feature selection to identify the most significant predictors of cancer diagnosis.
- Add cross-validation to ensure robust model performance.
- Include visualizations (e.g., correlation matrices, feature importance plots) to better understand the data.


## Contact
For questions or contributions, please open an issue or submit a pull request on GitHub.
