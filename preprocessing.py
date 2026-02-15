import pandas as pd
import re
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import os

# Create "plots" directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# 1. Initialize NLP tools
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 2. Load the Master Dataset
df = pd.read_csv("final_master_dataset_v2.csv")

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text) # Remove emails
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-letters
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    return " ".join(cleaned)

print("--- Step 1: Cleaning and Normalizing Text ---")
tqdm.pandas()
df['clean_text'] = df['text'].progress_apply(clean_text)

# --- NEW: Save the preprocessed master data ---
df.to_csv("dataset_preprocessed.csv", index=False)
print("✅ Saved 'dataset_preprocessed.csv'")

# --- NEW: Generate "Before vs After" Comparison Image ---
def create_comparison_img(df):
    sample_idx = 0 # Change index to see different samples
    original = df['text'].iloc[sample_idx][:300] + "..."
    cleaned = df['clean_text'].iloc[sample_idx][:300] + "..."
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    text_content = f"BEFORE PREPROCESSING:\n{original}\n\n" + "-"*50 + \
                   f"\n\nAFTER PREPROCESSING:\n{cleaned}"
    ax.text(0.05, 0.5, text_content, wrap=True, fontsize=10, family='monospace', verticalalignment='center')
    plt.title("Text Transformation: Raw vs. Cleaned", fontsize=14, fontweight='bold')
    plt.savefig('plots/preprocessing_comparison.png', bbox_inches='tight', dpi=300)
    print("✅ Generated 'preprocessing_comparison.png'")

create_comparison_img(df)

# 3. Split and Vectorize
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("\n--- Step 2: Generating Feature Representations ---")
tfidf_vec = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vec.fit_transform(X_train_raw)
X_test_tfidf = tfidf_vec.transform(X_test_raw)

# --- NEW: Generate Top Features Visualization ---
feature_names = tfidf_vec.get_feature_names_out()
def save_top_features(X, y, features):
    # Convert sparse matrix to mean importance
    pol_idx = [i for i, val in enumerate(y) if val == 'Politics']
    spr_idx = [i for i, val in enumerate(y) if val == 'Sport']
    
    pol_weights = X[pol_idx].mean(axis=0).A1
    spr_weights = X[spr_idx].mean(axis=0).A1
    
    feat_df = pd.DataFrame({'word': features, 'Politics': pol_weights, 'Sport': spr_weights})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    sns.barplot(data=feat_df.sort_values('Politics', ascending=False).head(15), x='Politics', y='word', ax=ax1, palette='Reds_r')
    sns.barplot(data=feat_df.sort_values('Sport', ascending=False).head(15), x='Sport', y='word', ax=ax2, palette='Blues_r')
    ax1.set_title('Politics Keywords'); ax2.set_title('Sport Keywords')
    plt.tight_layout()
    plt.savefig('plots/top_features.png')
    print("✅ Generated 'top_features.png'")

save_top_features(X_train_tfidf, y_train, feature_names)

# Final Exports (BoW and Ngrams)
bow_vec = CountVectorizer(max_features=5000); X_train_bow = bow_vec.fit_transform(X_train_raw); X_test_bow = bow_vec.transform(X_test_raw)
ngram_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)); X_train_ngram = ngram_vec.fit_transform(X_train_raw); X_test_ngram = ngram_vec.transform(X_test_raw)

features_out = {'y_train': y_train, 'y_test': y_test, 'bow': (X_train_bow, X_test_bow), 'tfidf': (X_train_tfidf, X_test_tfidf), 'ngram': (X_train_ngram, X_test_ngram)}
with open("features.pkl", "wb") as f: pickle.dump(features_out, f)
print(f"✅ SUCCESS! Feature matrices saved to 'features.pkl'")