
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Set global style for professional look
sns.set_style("whitegrid")
plt.rcParams.update({'figure.autolayout': True})

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    # Using a clean, professional color map
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                cbar=False, linewidths=1.5, linecolor='white', 
                annot_kws={"size": 14, "weight": "bold"}, square=True)
    
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14, pad=15, fontweight='bold', color="#333333")
    
    # Clean up ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    return fig

def plot_roc_curve(y_true, y_score, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plotting the curve with a professional color and thickness
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="#1f77b4", lw=2.5)
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", alpha=0.8)
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_title(f"{model_name} - ROC Analysis", fontsize=14, pad=15, fontweight='bold', color="#333333")
    
    # Improved legend
    ax.legend(loc="lower right", frameon=True, fontsize=10, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig

def plot_feature_importance(importances, feature_names, model_name="Random Forest"):
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Fixed deprecation warning by assigning hue to x (or y) and legend=False
    sns.barplot(
    x="Importance",
    y="Feature",
    data=df_imp,
    ax=ax,
    palette="viridis"
)

    
    ax.set_title(f"Feature Importance - {model_name}", fontsize=14, pad=15, fontweight='bold', color="#333333")
    ax.set_xlabel("Importance Score", fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=12, labelpad=10, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Remove top and right spines for a cleaner look
    sns.despine()
    
    return fig
