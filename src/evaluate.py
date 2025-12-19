from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , roc_auc_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, probs))        