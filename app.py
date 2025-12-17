import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.visualize import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'heart.csv')

# -------------------- Load Models & Data --------------------

@st.cache_resource
def load_resources():
    try:
        models = {
            "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')),
            "Random Forest": joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl')),
            "SVM": joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
        }
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
        return None

models, scaler = load_resources()
df = load_dataset()

if models is None or df is None:
    st.stop()

feature_names = df.drop("target", axis=1).columns.tolist()

# -------------------- Sidebar Controls --------------------

st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_choice]

st.title("Heart Disease Prediction Dashboard")
st.markdown("Provide patient details below to assess the risk of heart disease.")

# -------------------- User Input --------------------

def get_user_input(df, feature_names):
    input_data = {}
    
    col1, col2, col3 = st.columns(3)
    
    # Distribute fields across columns
    fields = feature_names
    
    for i, feature in enumerate(fields):
        # Determine column
        if i % 3 == 0:
            c = col1
        elif i % 3 == 1:
            c = col2
        else:
            c = col3
            
        with c:
            median_val = df[feature].median()

            if feature in ["age", "trestbps", "chol", "thalach"]:
                val = st.number_input(
                    f"{feature.replace('_', ' ').title()}", 
                    min_value=int(df[feature].min()), 
                    max_value=int(df[feature].max()),
                    value=int(median_val), 
                    step=1
                )
            elif feature == "sex":
                val = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female", index=int(median_val))
            elif feature == "fbs":
                val = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False", index=int(median_val))
            elif feature == "exang":
                val = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(median_val))
            elif feature == "cp":
                val = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], index=int(median_val), help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
            elif feature == "restecg":
                val = st.selectbox("Resting ECG Results", options=[0, 1, 2], index=int(median_val))
            elif feature == "slope":
                val = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2], index=int(median_val))
            elif feature == "ca":
                val = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3], index=int(median_val))
            elif feature == "thal":
                val = st.selectbox("Thalassemia", options=[0, 1, 2, 3], index=int(median_val))
            elif feature == "oldpeak":
                val = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=float(df[feature].max()), value=float(median_val), step=0.1)
            else:
                val = st.number_input(f"{feature.title()}", value=float(median_val))
            
            input_data[feature] = val

    return pd.DataFrame([input_data])

user_input_df = get_user_input(df, feature_names)

# -------------------- Prediction --------------------

if st.button("Predict", type="primary"):
    with st.spinner("Analyzing..."):
        try:
            input_scaled = scaler.transform(user_input_df)
            prediction = model.predict(input_scaled)[0]
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_scaled)[0][1]
            else:
                # Use decision function and normalize if needed, or just skip probability
                prob = None

            st.divider()
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("High Risk of Heart Disease")
                else:
                    st.success("Low Risk of Heart Disease")
            
            if prob is not None:
                with res_col2:
                    st.subheader("Probability")
                    st.metric(label="Risk Probability", value=f"{prob:.2%}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# -------------------- Model Evaluation --------------------

with st.expander("Show Model Performance Details"):
    st.subheader(f"Performance Metrics: {model_choice}")
    
    # Prepare test data (fixed split for consistency)
    from sklearn.model_selection import train_test_split
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Feature Importance
    if model_choice == "Random Forest":
        st.write("#### Feature Importance")
        fig_imp = plot_feature_importance(model.feature_importances_, feature_names, model_name=model_choice)
        st.pyplot(fig_imp)

    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        st.write("#### Confusion Matrix")
        fig_cm = plot_confusion_matrix(y_test, y_pred, model_name=model_choice)
        st.pyplot(fig_cm)
        
    with col_eval2:
        st.write("#### ROC Curve")
        try:
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_probs = model.decision_function(X_test_scaled)
            
            fig_roc = plot_roc_curve(y_test, y_probs, model_name=model_choice)
            st.pyplot(fig_roc)
        except Exception as e:
            st.info("ROC Curve not available for this model configuration.")

