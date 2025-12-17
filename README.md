# Heart Disease Prediction Model

This project provides a machine learning-based dashboard for predicting the risk of heart disease based on patient metrics. It utilizes Logistic Regression, Random Forest, and SVM models.

## Project Structure

- `app.py`: Main entry point for the Streamlit dashboard.
- `models/`: Directory containing pre-trained model files (.pkl).
- `data/`: Directory containing the dataset (`heart.csv`).
- `src/`: Source code for auxiliary functions (visualization, evaluation).
- `requirements.txt`: List of Python dependencies.

## Setup and Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the dashboard, you must use the `streamlit run` command. Do not run the script directly with Python.

```bash
streamlit run app.py
```

The dashboard will open in your default web browser (usually at `http://localhost:8501`).

### Troubleshooting

**"missing ScriptRunContext" Error:**
If you see a warning like `Thread 'MainThread': missing ScriptRunContext!`, it means you are trying to run the script with `python app.py`. Please use `streamlit run app.py` instead.
