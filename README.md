# Heart Disease Prediction Multi-Model Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project is an interactive **Heart Disease Prediction Dashboard** powered by Machine Learning. It allows users to input medical data (such as age, cholesterol levels, chest pain type, etc.) and receive a real-time risk assessment for heart disease.

The system utilizes three powerful classification algorithms to ensure reliable predictions:
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

Built with **Streamlit**, the application provides a user-friendly interface for both patients and healthcare professionals to explore risk factors and model performance metrics.

## 🚀 Key Features

- **Interactive Risk Prediction**: Easy-to-use side panel and form to input patient health metrics.
- **Multi-Model Support**: Switch between Logistic Regression, Random Forest, and SVM to compare predictions.
- **Real-time Probability**: Displays not just the classification (High/Low Risk) but also the confidence probability.
- **Visual Analytics**:
    - **Confusion Matrix**: To visualize model accuracy on test data.
    - **ROC Curve**: To analyze the trade-off between sensitivity and specificity.
    - **Feature Importance**: (Random Forest only) To understand which health factors contribute most to the prediction.
- **Responsive Design**: Clean and professional UI layout.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Backend Model Loading**: Joblib

## 📂 Project Structure

```
HeartDiseasePredictionMultiModel/
├── app.py                # Main Streamlit application entry point
├── requirements.txt      # Project dependencies
├── models/               # Pre-trained ML models (.pkl files)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl        # Data scaler for normalization
├── data/                 # Dataset directory
│   └── heart.csv         # Source dataset (Cleveland Heart Disease Data)
└── src/                  # Helper modules
    ├── visualize.py      # Plotting functions (ROC, Confusion Matrix)
    ├── evaluate.py       # Metrics evaluation
    └── preprocess.py     # Data cleaning pipelines
```

## ⚙️ Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/HeartDiseasePredictionMultiModel.git
cd HeartDiseasePredictionMultiModel
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run

To launch the dashboard, use the `streamlit run` command:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.


## 🏗️ Working & Process

### 1. Data Loading & Preprocessing
- Loads `heart.csv` dataset and splits into train/test sets.
- Features are scaled for model training and prediction.

### 2. Model Training (`train_v2.py`)
- Trains Logistic Regression, Random Forest, and SVM models with hyperparameter tuning.
- Builds an ensemble model for consensus prediction.
- Evaluates models (accuracy, confusion matrix, ROC AUC).
- Saves models and scaler in `models/`.

### 3. Web Application (`app.py` & `main.py`)
- `app.py`: Main Streamlit app for interactive prediction and analytics.
- `main.py`: CLI launcher for the Streamlit app (runs `app.py` via Python, useful for deployment or automation).
- Both run the same dashboard, but `main.py` is a wrapper for easier launching.

### 4. Prediction & Analytics
- User inputs patient data in the app.
- Selects model (Logistic Regression, Random Forest, SVM, Ensemble).
- App predicts risk and shows probability/confidence.
- Advanced analytics: confusion matrix, ROC curve, feature importance, EDA plots.

### 5. Visualization
- Professional plots for model diagnostics and data insights.
- Includes confusion matrix, ROC curve, feature importance, correlation heatmap, target distribution, and heart rate vs age scatter.

### 6. Evaluation
- Prints accuracy, confusion matrix, classification report, and ROC AUC for each model.

## ⚡ Difference: `app.py` vs `main.py`

- `app.py`: Directly run with `streamlit run app.py` for development and local use.
- `main.py`: Python script that launches the Streamlit app programmatically (useful for automation, deployment, or when you want to run with `python main.py`).
- Both show the same dashboard; `main.py` is just a convenience wrapper.

---

## ❓ One-Liner Questions & Answers

1. **What dataset is used for heart disease prediction?**  
   Cleveland Heart Disease Dataset (`data/heart.csv`).
2. **Which models are trained in this project?**  
   Logistic Regression, Random Forest, SVM, and Ensemble.
3. **What is the role of the ensemble model?**  
   Combines predictions from multiple models for better accuracy.
4. **How is data preprocessing handled?**  
   Features are scaled using StandardScaler in `preprocess.py`.
5. **Which Python library powers the web interface?**  
   Streamlit.
6. **What does the confusion matrix show?**  
   Model accuracy for healthy vs diseased predictions.
7. **How is feature importance visualized?**  
   Bar plot in the analytics tab (Random Forest only).
8. **What metric indicates model discrimination ability?**  
   ROC AUC score.
9. **How are models saved for later use?**  
   Using Joblib in the `models/` folder.
10. **What does the ROC curve represent?**  
    Trade-off between true positive and false positive rates.
11. **How is user input collected in the app?**  
    Through interactive form fields in Streamlit.
12. **What is the purpose of the scaler?**  
    Normalizes features for model training and prediction.
13. **Which file contains model training logic?**  
    `train_v2.py` and `src/model.py`.
14. **How is model evaluation performed?**  
    Using accuracy, confusion matrix, classification report, ROC AUC.
15. **What does the correlation heatmap display?**  
    Relationships between clinical features.
16. **How can you launch the web app?**  
    `streamlit run app.py` or `python main.py`.
17. **What does the target variable represent?**  
    Presence (1) or absence (0) of heart disease.
18. **How is hyperparameter tuning performed?**  
    GridSearchCV/RandomizedSearchCV in model training functions.
19. **What is the significance of the `main.py` file?**  
    Provides a Pythonic way to launch the Streamlit app.
20. **How are advanced analytics accessed in the app?**  
    Via the "Advanced Analytics & Data Insights" expander/tab in the dashboard.

---
