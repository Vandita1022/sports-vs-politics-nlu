import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# Create "plots" directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# 1. Load data and setup
try:
    df = pd.read_csv("full_evaluation_metrics.csv")
    with open("features.pkl", "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: Ensure 'full_evaluation_metrics.csv' and 'features.pkl' exist.")
    exit()

y_test_numeric = data['y_test'].map({'Politics': 0, 'Sport': 1})
X_test_tfidf = data['tfidf'][1]
X_test_bow = data['bow'][1]

# 2. PLOT A: Metric Comparison Bar Chart (TF-IDF)
plt.figure(figsize=(14, 7))
sns.set_style("whitegrid")
tfidf_df = df[df['Feature Set'] == 'TF-IDF'].melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
sns.barplot(data=tfidf_df, x='Model', y='value', hue='variable', palette='viridis')
plt.title('Performance Metric Comparison (TF-IDF Set)', fontsize=15, fontweight='bold')
plt.ylim(0.7, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/metric_comparison_final.png', dpi=300)

# 3. PLOT B: Accuracy Heatmap
plt.figure(figsize=(10, 6))
pivot_acc = df.pivot(index="Model", columns="Feature Set", values="Accuracy")
sns.heatmap(pivot_acc, annot=True, cmap="YlGnBu", fmt=".4f")
plt.title('Accuracy Heatmap: Features vs. Models', fontsize=14)
plt.tight_layout()
plt.savefig('plots/accuracy_heatmap_final.png', dpi=300)

# 4. PLOT C: ROC & Precision-Recall Curves (Top 3 Models)
models_to_plot = {
    "Logistic Regression": "trained_models/LogisticRegression_TF-IDF.pkl",
    "Naive Bayes": "trained_models/NaiveBayes_TF-IDF.pkl",
    "Linear SVM": "trained_models/SVM_Linear_TF-IDF.pkl"
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for name, path in models_to_plot.items():
    with open(path, "rb") as f:
        model = pickle.load(f)
    scores = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_tfidf)
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test_numeric, scores)
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.4f})')
    # PR
    prec, rec, _ = precision_recall_curve(y_test_numeric, scores)
    ax2.plot(rec, prec, label=f'{name} (AP = {average_precision_score(y_test_numeric, scores):.4f})')

ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title('ROC Curves', fontsize=14, fontweight='bold'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.legend()
ax2.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.legend()
plt.tight_layout()
plt.savefig('plots/advanced_curves.png', dpi=300)

# 5. PLOT D: Best vs. Worst Confusion Matrix
with open("trained_models/LogisticRegression_TF-IDF.pkl", "rb") as f: best_mod = pickle.load(f)
with open("trained_models/KNN_BoW.pkl", "rb") as f: worst_mod = pickle.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(confusion_matrix(data['y_test'], best_mod.predict(X_test_tfidf)), annot=True, fmt='d', cmap='Greens', ax=ax1, xticklabels=['Pol', 'Spr'], yticklabels=['Pol', 'Spr'])
ax1.set_title('BEST: LogReg (TF-IDF)\nAccuracy: 96.08%')
sns.heatmap(confusion_matrix(data['y_test'], worst_mod.predict(X_test_bow)), annot=True, fmt='d', cmap='Reds', ax=ax2, xticklabels=['Pol', 'Spr'], yticklabels=['Pol', 'Spr'])
ax2.set_title('WORST: KNN (BoW)\nAccuracy: 71.74%')
plt.tight_layout()
plt.savefig('plots/best_vs_worst_confusion.png', dpi=300)

plt.show()
print("✅ Final robust visual package generated.")