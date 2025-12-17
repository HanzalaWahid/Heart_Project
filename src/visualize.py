
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
	cm = confusion_matrix(y_true, y_pred)
	fig, ax = plt.subplots(figsize=(5, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, linewidths=0.5, linecolor='gray', square=True)
	ax.set_xlabel("Predicted", fontsize=12)
	ax.set_ylabel("Actual", fontsize=12)
	ax.set_title(f"{model_name} Confusion Matrix", fontsize=14)
	plt.tight_layout()
	return fig

def plot_roc_curve(y_true, y_score, model_name="Model"):
	fpr, tpr, _ = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)
	fig, ax = plt.subplots(figsize=(5, 4))
	ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="navy", lw=2)
	ax.plot([0, 1], [0, 1], 'r--', lw=1)
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel("False Positive Rate", fontsize=12)
	ax.set_ylabel("True Positive Rate", fontsize=12)
	ax.set_title(f"{model_name} ROC Curve", fontsize=14)
	ax.legend(loc="lower right")
	plt.tight_layout()
	return fig

def plot_feature_importance(importances, feature_names, model_name="Random Forest"):
	df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
	df_imp = df_imp.sort_values(by="Importance", ascending=False)
	fig, ax = plt.subplots(figsize=(8, 5))
	sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, palette="viridis")
	ax.set_title(f"{model_name} Feature Importance", fontsize=14)
	ax.set_xlabel("Importance", fontsize=12)
	ax.set_ylabel("Feature", fontsize=12)
	plt.tight_layout()
	return fig
