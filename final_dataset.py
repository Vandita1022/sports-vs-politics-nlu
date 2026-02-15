import pandas as pd
import hashlib
from tqdm import tqdm

# Files
EXISTING_DATA = "final_master_dataset.csv"
NEW_DATA_CSV = "bbc-text.csv"
OUTPUT_FILE = "final_master_dataset_v2.csv"

def get_hash(text):
    """Generates a unique fingerprint for text."""
    # Normalize text (lowercase + strip) to ensure robust duplicate detection
    text = str(text).lower().strip()
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def merge_bbc_archive():
    print("--- 🛡️ Starting Strict Merge (BBC Archive -> Master Dataset) ---")

    # 1. Load Existing Master Data
    if not os.path.exists(EXISTING_DATA):
        print(f"❌ Error: Could not find '{EXISTING_DATA}'.")
        return

    df_master = pd.read_csv(EXISTING_DATA)
    print(f"Loaded Master Dataset: {len(df_master)} rows")
    
    # Build a "Fingerprint Set" of what you already have
    # This makes checking for duplicates instant (O(1) complexity)
    existing_hashes = set(df_master['text'].apply(get_hash))
    print(f"   -> Indexed {len(existing_hashes)} unique existing articles.")

    # 2. Load and Process New BBC Data
    if not os.path.exists(NEW_DATA_CSV):
        print(f"❌ Error: Could not find '{NEW_DATA_CSV}'.")
        return

    df_bbc = pd.read_csv(NEW_DATA_CSV)
    # Filter only relevant categories
    df_bbc = df_bbc[df_bbc['category'].isin(['sport', 'politics'])]
    print(f"Loaded BBC Archive (Raw): {len(df_bbc)} rows")

    new_rows = []
    added_count = 0
    duplicate_count = 0

    # 3. Iterate and Deduplicate
    print("\nChecking for duplicates...")
    for index, row in tqdm(df_bbc.iterrows(), total=len(df_bbc)):
        text = row['text']
        text_hash = get_hash(text)

        # STRICT CHECK: If we have seen this text before, SKIP IT
        if text_hash in existing_hashes:
            duplicate_count += 1
            continue
        
        # If unique, prepare it for addition
        label = 'Sport' if row['category'] == 'sport' else 'Politics'
        
        new_rows.append({
            'text': text,
            'label': label,
            'source': 'BBC Archive (2005)'
        })
        
        # Add hash to set (in case BBC archive itself has internal duplicates)
        existing_hashes.add(text_hash)
        added_count += 1

    # 4. Merge and Save
    if added_count > 0:
        df_new = pd.DataFrame(new_rows)
        df_final = pd.concat([df_master, df_new], ignore_index=True)
        
        # Shuffle
        df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
        
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ SUCCESS! Added {added_count} new unique articles.")
        print(f"🚫 Skipped {duplicate_count} duplicates.")
        print(f"📊 New Total Size: {len(df_final)}")
        print(f"Saved to: {OUTPUT_FILE}")
        
        # Breakdown
        print("\nFinal Composition:")
        print(df_final.groupby(['source', 'label']).size())
        
    else:
        print("\n⚠️ No new unique data found. Master dataset remains unchanged.")

if __name__ == "__main__":
    import os
    merge_bbc_archive()