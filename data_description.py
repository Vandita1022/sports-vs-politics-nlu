import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create "plots" directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Load the master dataset
df = pd.read_csv("final_master_dataset_v2.csv")

# Set the visual style
sns.set_theme(style="whitegrid")

# 1. Plot Class Distribution (Sport vs Politics)
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='label', palette='magma')
plt.title('Final Class Distribution (Sports vs. Politics)', fontsize=14)
plt.savefig('plots/class_dist.png', dpi=300)
plt.show()

# 2. Plot Source Composition (Stacked Bar)
source_counts = df.groupby(['source', 'label']).size().unstack()
source_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#3498db', '#e74c3c'])
plt.title('Article Contribution by Data Source', fontsize=14)
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/source_comp.png', dpi=300)
plt.show()

# 3. Word Count Distribution (Density Plot)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='word_count', hue='label', fill=True, common_norm=False, palette='crest')
plt.title('Distribution of Article Word Counts', fontsize=14)
plt.xlim(0, 1500) # Zoom in for clarity
plt.savefig('plots/word_dist.png', dpi=300)
plt.show()