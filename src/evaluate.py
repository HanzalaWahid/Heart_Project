from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , roc_auc_score

def evaluate_model(model , X_test , Y_test):
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:,1]
    except:
        pass

    print("Accuracy:" , accuracy_score(Y_test,preds))
    print("Confusion Matrix:\n" , confusion_matrix(Y_test , preds))
    print("Classification Report:\n" , classification_report(Y_test , preds))

    if probs is not None:
        roc_auc = roc_auc_score(Y_test , probs)
        print("ROC AUC Score:" , roc_auc)
        