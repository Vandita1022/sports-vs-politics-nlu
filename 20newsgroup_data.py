import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm  # Progress bar

# Configuration
LOCAL_DATA_DIR = "sports_politics_data"
OUTPUT_FILE = "final_master_dataset.csv"

def load_local_data(root_dir):
    """Loads text files from your local sports/politics folders."""
    data = []
    categories = ['sports', 'politics']
    
    print(f"--- 📂 Loading Local Data from '{root_dir}' ---")
    
    # Iterate over both category folders
    for category in categories:
        folder_path = os.path.join(root_dir, category)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder '{folder_path}' not found.")
            continue
            
        files = os.listdir(folder_path)
        # Filter for .txt files only
        files = [f for f in files if f.endswith('.txt')]
        
        print(f"Processing '{category}' folder ({len(files)} files)...")
        
        # TQDM Loop for progress bar
        for filename in tqdm(files, desc=f"Reading {category}", unit="file"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                    # Determine Source (BBC vs Kaggle) based on filename
                    if "kaggle" in filename.lower():
                        source = "HuffPost (Kaggle)"
                    else:
                        source = "BBC (Scraped)"
                    
                    # Standardize Label: 'Sport' vs 'Politics'
                    label = 'Sport' if category == 'sports' else 'Politics'
                    
                    # Basic length check
                    if len(text.strip()) > 10:
                        data.append({
                            'text': text, 
                            'label': label,
                            'source': source
                        })
            except Exception as e:
                # Silently skip bad files
                continue
                
    return pd.DataFrame(data)

def load_20newsgroups():
    """Fetches and processes the standard 20 Newsgroups dataset."""
    print(f"\n--- 🌐 Loading 20 Newsgroups Dataset ---")
    
    # Define categories to fetch
    categories = [
        'rec.sport.hockey', 'rec.sport.baseball',  # Sports
        'talk.politics.mideast', 'talk.politics.guns', 'talk.politics.misc' # Politics
    ]
    
    # Fetch data (remove headers/footers to prevent cheating)
    # This might take a few seconds if not cached
    dataset = fetch_20newsgroups(subset='all', categories=categories, 
                                 remove=('headers', 'footers', 'quotes'))
    
    data = []
    total_files = len(dataset.data)
    
    # TQDM Loop
    for i in tqdm(range(total_files), desc="Processing Newsgroups", unit="doc"):
        text = dataset.data[i]
        target_idx = dataset.target[i]
        target_name = dataset.target_names[target_idx]
        
        # Map to Binary Labels
        if 'sport' in target_name:
            label = 'Sport'
        else:
            label = 'Politics'
            
        if len(text.strip()) > 50:  # Skip empty/short messages
            data.append({
                'text': text,
                'label': label,
                'source': '20 Newsgroups'
            })
            
    return pd.DataFrame(data)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Load Local (Scraped + Kaggle)
    df_local = load_local_data(LOCAL_DATA_DIR)
    
    # 2. Load Standard (20 Newsgroups)
    df_standard = load_20newsgroups()
    
    # 3. Merge
    print(f"\n--- 🔄 Merging & Cleaning ---")
    df_final = pd.concat([df_local, df_standard], ignore_index=True)
    
    print(f"Raw Count: {len(df_final)}")
    
    # 4. Deduplicate (Remove exact text matches)
    initial_len = len(df_final)
    df_final.drop_duplicates(subset='text', inplace=True)
    print(f"Removed {initial_len - len(df_final)} duplicates.")
    
    # 5. Shuffle (Randomize order)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 6. Save
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    # 7. Final Report
    print(f"\n✅ SUCCESS! Master Dataset saved as '{OUTPUT_FILE}'")
    print("\n--- Final Composition ---")
    print(df_final.groupby(['source', 'label']).size())
    print(f"\nTotal Samples: {len(df_final)}")