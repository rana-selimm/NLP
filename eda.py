import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Paths
cleaned_csv_path = "cleaned_qa"
cleaned_txt_path = "cleaned"

# 1. Load all CSVs into one big table
csv_files = [f for f in os.listdir(cleaned_csv_path) if f.endswith('.csv')]
all_dfs = [pd.read_csv(os.path.join(cleaned_csv_path, f)) for f in csv_files]
df = pd.concat(all_dfs)

# 2. Analyze Lengths
df['q_len'] = df['question'].astype(str).apply(lambda x: len(x.split()))
df['a_len'] = df['answer'].astype(str).apply(lambda x: len(x.split()))

print(f"Total QA Pairs: {len(df)}")
print(f"Average Question Length: {df['q_len'].mean():.2f} words")
print(f"Average Answer Length: {df['a_len'].mean():.2f} words")

# 3. Plot Question Length Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['q_len'], bins=20, kde=True, color='skyblue')
plt.title('Question Length Distribution')
plt.xlabel('Words')
plt.savefig('q_dist.png')

# 4. Top 20 Most Frequent Words
all_words = " ".join(df['question'].astype(str)).split()
word_counts = Counter(all_words).most_common(20)
words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])

plt.figure(figsize=(12, 8))
sns.barplot(data=words_df, x='Count', y='Word', palette='viridis')
plt.title('Top 20 Frequent Words in Questions')
plt.savefig('top_words.png')

print("Charts saved as q_dist.png and top_words.png!")

# 5. Build Vocabulary Index
unique_words = sorted(list(set(all_words)))
word_to_idx = {word: i for i, word in enumerate(unique_words)}

# Save a sample of the index to show in your report
with open("vocab_index_sample.txt", "w", encoding="utf-8") as f:
    for i, (word, idx) in enumerate(word_to_idx.items()):
        if i > 50: break # Just save the first 50
        f.write(f"{word}: {idx}\n")

print(f"Vocabulary Size: {len(unique_words)}")