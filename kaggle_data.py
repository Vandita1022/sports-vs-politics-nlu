import json
import os

# Configuration
KAGGLE_FILE = "News_Category_Dataset_v3.json"
BASE_DIR = "sports_politics_data"
LIMIT_PER_CATEGORY = 1000  # Limits the number of files so your folder doesn't explode

def convert_json_to_txt():
    # specific mappings for HuffPost categories to your folder names
    category_map = {
        "SPORTS": "sports",
        "POLITICS": "politics"
    }
    
    # Counters to keep track of how many we've saved
    counts = {"SPORTS": 0, "POLITICS": 0}
    
    print(f"--- Converting {KAGGLE_FILE} to .txt format ---")
    
    try:
        with open(KAGGLE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse the JSON line
                article = json.loads(line)
                cat = article.get('category')
                
                # Only process if it's Sports or Politics
                if cat in category_map:
                    # Check if we hit the limit
                    if counts[cat] >= LIMIT_PER_CATEGORY:
                        continue
                        
                    # Create the text content: Headline + Description
                    content = f"{article['headline']}\n{article['short_description']}"
                    
                    # Define the output filename (e.g., kaggle_sports_001.txt)
                    folder_name = category_map[cat]
                    filename = f"kaggle_{folder_name}_{counts[cat]+1:04}.txt"
                    save_path = os.path.join(BASE_DIR, folder_name, filename)
                    
                    # Ensure directory exists (just in case)
                    os.makedirs(os.path.join(BASE_DIR, folder_name), exist_ok=True)
                    
                    # Write to .txt file
                    with open(save_path, "w", encoding="utf-8") as out_file:
                        out_file.write(content)
                    
                    counts[cat] += 1
                    
                # Stop if both categories are full
                if counts["SPORTS"] >= LIMIT_PER_CATEGORY and counts["POLITICS"] >= LIMIT_PER_CATEGORY:
                    break
                    
        print(f"✅ Conversion Complete!")
        print(f"   - Added {counts['SPORTS']} Sports articles to '{BASE_DIR}/sports'")
        print(f"   - Added {counts['POLITICS']} Politics articles to '{BASE_DIR}/politics'")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{KAGGLE_FILE}'. Please check the filename.")

if __name__ == "__main__":
    convert_json_to_txt()