import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import sys
import io

# Force terminal output to UTF-8 to handle Arabic text without crashing
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. SETUP PATHS 
cleaned_csv_path = "cleaned_qa"
cleaned_txt_path = "cleaned" 

# 2. LOAD ALL DATA
csv_files = [f for f in os.listdir(cleaned_csv_path) if f.endswith('.csv')]
all_dfs = [pd.read_csv(os.path.join(cleaned_csv_path, f)) for f in csv_files]
df_qa = pd.concat(all_dfs)

transcript_texts = []
for txt_file in os.listdir(cleaned_txt_path):
    with open(os.path.join(cleaned_txt_path, txt_file), "r", encoding="utf-8") as f:
        transcript_texts.append(f.read())

# 3. TEXTUAL DISTRIBUTIONS
df_qa['q_len'] = df_qa['question'].astype(str).apply(lambda x: len(x.split()))
df_qa['a_len'] = df_qa['answer'].astype(str).apply(lambda x: len(x.split()))

print("--- DATASET OVERVIEW ---")
print(f"Total QA Pairs: {len(df_qa)}")
print(f"Avg Question Length: {df_qa['q_len'].mean():.2f} words")
print(f"Avg Answer Length: {df_qa['a_len'].mean():.2f} words")

# 4. VOCABULARY & LINGUISTIC ANALYSIS
all_text = " ".join(df_qa['question'].astype(str)) + " " + \
           " ".join(df_qa['answer'].astype(str)) + " " + \
           " ".join(transcript_texts)

tokens = all_text.split()
print(f"Final Vocabulary Size: {len(set(tokens))}")

english_words = [w for w in tokens if re.search(r'[a-zA-Z]', w)]
print(f"Code-Switching Percentage: {(len(english_words) / len(tokens)) * 100:.2f}%")

dialect_markers = ["مش", "اللي", "عشان", "ده", "دي", "ايه", "كده"]
dialect_stats = {word: tokens.count(word) for word in dialect_markers}
print(f"Dialect Marker Frequency: {dialect_stats}")

# 5. FIXED TRACEABILITY CHECK
print("\n--- TRACEABILITY CHECK ---")
# We try to match the first CSV to its corresponding TXT file
sample_csv_name = csv_files[0]
# Common pattern: removing '_QA' or 'cleaned_' to find the matching transcript
possible_txt_name = sample_csv_name.replace('.csv', '.txt').replace('_QA', '')

# Search for the file in the directory to avoid exact naming errors
txt_files = os.listdir(cleaned_txt_path)
match = [f for f in txt_files if sample_csv_name.split('_')[1] in f] # match by video ID

if match:
    with open(os.path.join(cleaned_txt_path, match[0]), "r", encoding="utf-8") as f:
        sample_transcript = f.read()
    
    sample_qa = pd.read_csv(os.path.join(cleaned_csv_path, sample_csv_name))
    traceable_count = sum(1 for ans in sample_qa['answer'].head(100) if str(ans) in sample_transcript)
    print(f"Traceability Rate for {match[0]}: {traceable_count}%")
else:
    print("Could not find matching transcript for traceability check. Check filenames.")

# 6. VISUALIZATIONS
plt.figure(figsize=(10, 6))
sns.histplot(df_qa['q_len'], bins=20, kde=True, color='skyblue')
plt.title('Question Word Count Distribution')
plt.savefig('q_dist.png')

arabic_tokens = [t for t in tokens if not re.search(r'[a-zA-Z]', t)]
top_20 = Counter(arabic_tokens).most_common(20)
words_df = pd.DataFrame(top_20, columns=['Word', 'Count'])

plt.figure(figsize=(12, 8))
sns.barplot(data=words_df, x='Count', y='Word', hue='Word', legend=False, palette='viridis')
plt.title('Top 20 Frequent Arabic Words')
plt.savefig('top_words.png')

print("\nEDA COMPLETE: Charts saved as q_dist.png and top_words.png")