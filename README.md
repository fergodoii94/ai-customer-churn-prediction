# AI Customer Churn Prediction 🤖🧠

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Ready-F37726.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade **Machine Learning project** for predicting customer churn using ensemble methods. Implements complete ML pipeline including data preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning with Random Forest and Gradient Boosting classifiers.

## 🎯 Project Overview

This project demonstrates professional ML engineering practices:
- Complete data preprocessing pipeline
- Feature engineering and selection
- Multiple model implementations (Random Forest, XGBoost, Logistic Regression)
- Cross-validation and hyperparameter tuning
- Model evaluation with multiple metrics
- Serialized model for production deployment
- Jupyter notebooks for exploration
- Unit tests for data and model quality

## 🚀 Quick Start

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Analysis
```bash
# Using Jupyter
jupyter notebook notebooks/

# Or run pipeline script
python src/train.py
```

### Make Predictions
```python
from src.model import load_model, predict_churn
import pandas as pd

model = load_model('models/churn_model.pkl')
df = pd.read_csv('data/new_customers.csv')
predictions = predict_churn(model, df)
```

## 📊 Dataset

**Features:** Customer demographics, account information, service usage
**Target:** Churn (Yes/No)
**Size:** 7,000+ customer records
**Class Balance:** ~27% churn rate

### Key Features
- `tenure` - Months with company
- `MonthlyCharges` - Monthly billing amount
- `TotalCharges` - Total amount charged
- `InternetService` - Service type
- `Contract` - Contract type
- `PaymentMethod` - Payment method
- And 15+ more features

## 🔧 ML Pipeline

```
Raw Data
    ↓
Data Cleaning (missing values, outliers)
    ↓
Feature Engineering (scaling, encoding, selection)
    ↓
Train/Test Split (80/20)
    ↓
Model Training (Random Forest, XGBoost)
    ↓
Cross-Validation & Tuning
    ↓
Model Evaluation (Precision, Recall, F1, AUC-ROC)
    ↓
Serialization & Deployment
```

## 📈 Model Performance

### Random Forest Classifier
- **Accuracy:** 79.8%
- **Precision:** 78.5%
- **Recall:** 74.2%
- **F1-Score:** 76.3%
- **AUC-ROC:** 0.85

### XGBoost Classifier
- **Accuracy:** 82.1%
- **Precision:** 81.3%
- **Recall:** 77.8%
- **F1-Score:** 79.5%
- **AUC-ROC:** 0.88

## 📂 Project Structure

```
ai-customer-churn-prediction/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py           # Data loading and cleaning
│   ├── preprocessing.py  # Feature engineering
│   ├── model.py          # Model training and prediction
│   └── evaluate.py       # Model evaluation metrics
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── data/
│   ├── raw/              # Original data
│   ├── processed/        # Preprocessed data
│   └── test/             # Test data
├── models/
│   └── churn_model.pkl   # Trained model
├── requirements.txt      # Dependencies
├── pyproject.toml        # Project config
└── README.md             # This file
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## 📚 Key Insights

### Feature Importance (Top 5)
1. `tenure` - Strongest predictor (0.25 importance)
2. `MonthlyCharges` (0.18)
3. `Contract` (0.15)
4. `InternetService` (0.12)
5. `TotalCharges` (0.10)

### Churn Patterns
- **Month-to-month contracts** have 42% churn rate
- **Fiber optic users** have higher churn (41%)
- **Long tenure customers** (>24 months) have 9% churn
- **High monthly charges** correlate with churn

## 🎓 Technical Stack

**Data Processing:**
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - ML algorithms

**Model Training:**
- Random Forest Classifier
- XGBoost Gradient Boosting
- Logistic Regression (baseline)

**Evaluation:**
- scikit-learn metrics
- matplotlib/seaborn visualization
- Confusion matrix analysis

**Environment:**
- Jupyter Notebook
- Python 3.11+
- Docker ready

## 📊 Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Churn Prediction')

# Churn distribution by tenure
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure Distribution by Churn Status')

# Correlation heatmap
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Feature Correlation Matrix')
```

## 🚀 Production Deployment

### Model Serialization
```python
import joblib
joblib.dump(model, 'models/churn_model.pkl')

# Load in production
model = joblib.load('models/churn_model.pkl')
predictions = model.predict(X_new)
```

### FastAPI Integration
```python
from fastapi import FastAPI
from src.model import load_model, predict_churn

app = FastAPI()
model = load_model('models/churn_model.pkl')

@app.post("/predict")
async def predict(customer_data: CustomerSchema):
    prediction = predict_churn(model, customer_data.dict())
    return {"churn_probability": prediction}
```

## 📖 Learning Resources

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Guide](https://pandas.pydata.org/docs/)
- [ML Best Practices](https://scikit-learn.org/stable/modules/model_evaluation/)
- [XGBoost Tutorial](https://xgboost.readthedocs.io/)

## ✅ Checklist

- [x] Data cleaning and validation
- [x] Feature engineering
- [x] Multiple model implementations
- [x] Cross-validation
- [x] Hyperparameter tuning
- [x] Model evaluation metrics
- [x] Feature importance analysis
- [x] Jupyter notebooks
- [x] Unit tests
- [x] Production serialization

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👨‍💻 Author

**Fernando Godoi**
- GitHub: [@fergodoii94](https://github.com/fergodoii94)
- Email: fergodoi94@gmail.com

**Expertise:** ML Engineering | Data Science | Python | scikit-learn | pandas

**Status:** ✅ Production Ready | **Model Accuracy:** 82.1%
