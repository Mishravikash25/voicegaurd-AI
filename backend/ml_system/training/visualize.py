import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Ensure we can import from the ml_system root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_system.config import BASE_DIR

# Custom Visualizations Directory
VISUALS_DIR = BASE_DIR / "visualizations"
os.makedirs(VISUALS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Aesthetic Settings for matplotlib/seaborn
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#0f172a",          # slate-950
    "figure.facecolor": "#0f172a",
    "grid.color": "#1e293b",              # slate-800
    "text.color": "#f8fafc",              # slate-50
    "axes.labelcolor": "#cbd5e1",         # slate-300
    "xtick.color": "#94a3b8",             # slate-400
    "ytick.color": "#94a3b8"
})

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    Plots and saves a styled Confusion Matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Custom neon indigo color map
    cmap = sns.dark_palette("#6366f1", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=["Genuine", "Fake"], 
                yticklabels=["Genuine", "Fake"],
                annot_kws={"size": 16, "weight": "bold"})
                
    plt.title('Audio Forensic Confusion Matrix', fontsize=18, fontweight='bold', color='white', pad=20)
    plt.ylabel('Actual Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    
    save_path = os.path.join(VISUALS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"Saved: {save_path}")

def plot_roc_curve(y_true, y_probs, filename="roc_curve.png"):
    """
    Plots and saves the Receiver Operating Characteristic (ROC) curve.
    """
    # y_probs is expected to be the probability of the positive class (i.e., Fake/1)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#6366f1', lw=3, label=f'Forensic SVM (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#ef4444', lw=2, linestyle='--', alpha=0.5, label='Random Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold', color='white', pad=20)
    plt.legend(loc="lower right", facecolor='#0f172a', edgecolor='#1e293b')
    
    save_path = os.path.join(VISUALS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"Saved: {save_path}")

def plot_feature_importance(model, feature_names=None, filename="feature_importance.png"):
    """
    Plots the absolute values of the coefficients of a linear model to show feature importance.
    Note: Only works for linear kernels (Linear SVM or Logistic Regression).
    """
    if not hasattr(model, 'coef_'):
        logger.warning("Model does not have 'coef_' attribute. Ensure it is a linear model.")
        logger.warning("Skipping Feature Importance plot (using RBF/Poly SVM?).")
        return

    # Extract absolute coefficients (importance)
    importance = np.abs(model.coef_[0])
    
    # If feature names aren't provided, generate generic ones
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(len(importance))]
        
    # Aggregate to top 20 features for readability if too many
    if len(importance) > 20:
        indices = np.argsort(importance)[-20:]
        importance = importance[indices]
        feature_names = [feature_names[i] for i in indices]
    else:
        indices = np.argsort(importance)
        importance = importance[indices]
        feature_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 8))
    
    # Horizontal bar plot
    bars = plt.barh(range(len(importance)), importance, align='center', color='#6366f1', alpha=0.8)
    plt.yticks(range(len(importance)), feature_names, fontsize=10)
    
    plt.xlabel('Absolute Coefficient Magnitude', fontsize=12)
    plt.title('Top 20 Acoustic Feature Importances', fontsize=16, fontweight='bold', color='white', pad=20)
    
    save_path = os.path.join(VISUALS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"Saved: {save_path}")

def plot_score_distribution(y_true, y_probs, filename="score_distribution.png"):
    """
    Plots overlapping density distributions of predicted probabilities 
    for Genuine vs Fake audio.
    """
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by true class
    genuine_probs = [p for t, p in zip(y_true, y_probs) if t == 0]
    fake_probs = [p for t, p in zip(y_true, y_probs) if t == 1]
    
    # Plot densities
    sns.kdeplot(genuine_probs, color='#22c55e', fill=True, alpha=0.3, linewidth=2, label='Genuine Audio')
    sns.kdeplot(fake_probs, color='#ef4444', fill=True, alpha=0.3, linewidth=2, label='Synthetic/Fake Audio')
    
    plt.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, label='Decision Threshold')
    
    plt.xlim(0, 1)
    plt.xlabel('Predicted Fraud Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Forensic Score Distribution', fontsize=16, fontweight='bold', color='white', pad=20)
    plt.legend(loc="upper right", facecolor='#0f172a', edgecolor='#1e293b')
    
    save_path = os.path.join(VISUALS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    logger.info(f"Saved: {save_path}")

def generate_all_visuals(model, X_test_scaled, y_test, feature_names=None):
    """
    Runs inference on the test set and generates all visualizations sequentially.
    """
    logger.info("Generating Model Visualizations...")
    
    # Get predictions
    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)[:, 1] # Probability of being Fake (class 1)
    
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_probs)
    plot_score_distribution(y_test, y_probs)
    
    # Feature importance only if linear model
    if hasattr(model, 'coef_'):
        plot_feature_importance(model, feature_names)
        
    logger.info("All visualizations successfully generated and saved to 'ml_system/visualizations/'")

if __name__ == "__main__":
    # Mock data execution block for quick visual testing
    logger.info("Running syntax and mock generation test...")
    np.random.seed(42)
    mock_y_true = np.random.randint(0, 2, 100)
    mock_y_probs = np.where(mock_y_true == 1, 
                            np.clip(np.random.normal(0.8, 0.15, 100), 0, 1), 
                            np.clip(np.random.normal(0.2, 0.15, 100), 0, 1))
    mock_y_pred = (mock_y_probs > 0.5).astype(int)
    
    plot_confusion_matrix(mock_y_true, mock_y_pred, filename="test_cm.png")
    plot_roc_curve(mock_y_true, mock_y_probs, filename="test_roc.png")
    plot_score_distribution(mock_y_true, mock_y_probs, filename="test_dist.png")
    
    logger.info("Mock tests passed.")
