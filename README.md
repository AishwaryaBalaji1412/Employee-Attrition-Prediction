# Employee Attrition Prediction

A machine learning project that predicts employee attrition using HR data. It compares multiple classification models, evaluates their performance, and uses SHAP to explain important prediction factors. A Streamlit application allows users to make predictions using employee data.

## Project Overview

Employee attrition refers to employees leaving an organization. This project explores employee-related factors associated with attrition and builds machine learning models to estimate the likelihood of an employee leaving.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and categorical encoding
- Class imbalance handling using SMOTENC
- Machine learning model training and hyperparameter tuning
- Model evaluation and threshold optimization
- Model explainability using SHAP
- Interactive predictions through Streamlit

## Dataset

- **Dataset:** IBM HR Analytics Employee Attrition & Performance
- **Total records:** 1,470
- **Original features:** 35
- **Target variable:** Attrition (Yes/No)

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Imbalanced Data Handling:** Imbalanced-learn (SMOTENC)
- **Model Explainability:** SHAP
- **Application:** Streamlit

## Machine Learning Models

Three approaches are implemented and compared:

1. Random Forest Classifier
2. XGBoost Classifier
3. Soft Voting Ensemble

GridSearchCV with stratified cross-validation is used for hyperparameter tuning. Classification thresholds are also optimized using out-of-fold predicted probabilities.

## Model Evaluation

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### Test Set Results

| Model | Threshold | F1-score | Precision | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Random Forest | Default | 0.4944 | 0.5238 | 0.4681 | 0.8227 |
| Random Forest | Tuned | 0.4615 | 0.3614 | 0.6383 | 0.8227 |
| XGBoost | Default | 0.4792 | 0.4694 | 0.4894 | 0.7998 |
| XGBoost | Tuned | 0.4516 | 0.3636 | 0.5957 | 0.7998 |
| Ensemble | Default | 0.4731 | 0.4783 | 0.4681 | 0.8165 |
| Ensemble | Tuned | 0.4948 | 0.4800 | 0.5106 | 0.8165 |

The tuned ensemble achieved the highest F1-score among the evaluated configurations. Random Forest achieved the highest ROC-AUC.

Threshold tuning changes the balance between precision and recall. For example, the tuned Random Forest identifies more employees in the attrition class but also produces more false positives.

## Model Explainability

SHAP (SHapley Additive exPlanations) is used to interpret the Random Forest and XGBoost models and examine which features influence their predictions.

Generated explanations include:
- SHAP summary bar plots
- SHAP summary dot plots

## Streamlit Application

The application supports:
- Employee attrition prediction
- Batch predictions using CSV uploads
- Selection between trained models
- Display of prediction probabilities
- Exploratory analysis and model explanation visualizations

A sample input file is included as `sample_employees.csv`.

## Project Structure

```text
employee-attrition-project/
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
├── README.md
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── ensemble_model.pkl
├── outputs/
│   ├── metrics/
│   │   └── thresholds.json
│   └── plots/
├── sample_employees.csv
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── exploratory_analysis.py
    ├── model_training.py
    ├── model_evaluation.py
    ├── model_explainability.py
    └── prediction.py
```

## Installation and Setup

Clone the repository and navigate to the project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Retrain the Models

To repeat preprocessing, model training, evaluation, and SHAP visualization generation:

```bash
python train.py
```

The trained models are saved in `models/`, evaluation plots in `outputs/plots/`, and optimized thresholds in `outputs/metrics/`.

## Limitations

- The dataset is relatively small and may not represent every organization.
- Employee attrition is influenced by factors that may not be captured in the available data.
- Model predictions are estimates, not certainties.
- This project is intended for educational and portfolio purposes, not as the sole basis for real employment decisions.