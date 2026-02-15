import pickle
import pandas as pd
import os
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Setup Environment
MODEL_SAVE_PATH = "trained_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 2. Load preprocessed features
with open("features.pkl", "rb") as f:
    data = pickle.load(f)

y_train, y_test = data['y_train'], data['y_test']

# 3. Define the 6 models
models_dict = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM_Linear": LinearSVC(dual=False),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

feature_sets = {
    "BoW": data['bow'],
    "TF-IDF": data['tfidf'],
    "N-Grams": data['ngram']
}

full_results = []

print("--- 🚀 Starting Multi-Model Evaluation Pipeline ---")

# 4. Training and Metrics Loop
total_iters = len(models_dict) * len(feature_sets)
pbar = tqdm(total=total_iters, desc="Overall Progress")

for model_name, model in models_dict.items():
    for feat_name, (X_train, X_test) in feature_sets.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict and Score
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Calculate Precision, Recall, F1 (weighted to handle any slight imbalances)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        full_results.append({
            "Model": model_name,
            "Feature Set": feat_name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
        # Save model
        with open(f"{MODEL_SAVE_PATH}/{model_name}_{feat_name}.pkl", 'wb') as f:
            pickle.dump(model, f)
            
        pbar.update(1)

pbar.close()

# 5. Export Results
df_res = pd.DataFrame(full_results)
df_res.to_csv("full_evaluation_metrics.csv", index=False)

print("\n--- Summary of Results ---")
print(df_res.sort_values(by="Accuracy", ascending=False).head(10))
print(f"\n✅ All results saved to 'full_evaluation_metrics.csv' and models to '{MODEL_SAVE_PATH}'")