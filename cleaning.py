import os
import re
import json
import pandas as pd

import sys
import io

# Force terminal output to use UTF-8 to handle Arabic text and symbols
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. SETUP PATHS
raw_path = "data"
txt_out = "cleaned"
csv_out = "cleaned_qa"

for folder in [txt_out, csv_out]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 2. THE CLEANING ENGINE
def normalize_arabic(text):
    """
    Standardizes Arabic characters and removes noise as per MS1 requirements.
    Addresses: Orthographic inconsistencies and dialectal variation.
    """
    if not isinstance(text, str) or text == 'nan': 
        return ""
    
    # Remove Tashkeel (vowels/diacritics) - [cite: 66, 67]
    tashkeel = re.compile(r'[\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)

    # Normalize Alif shapes (أ، إ، آ -> ا) - [cite: 43, 67]
    text = re.sub("[إأآ]", "ا", text)
    
    # Normalize final Ya and Ta Marbuta (ى -> ي, ة -> ه) - [cite: 41, 43]
    # Note: Using 'ه' for 'ة' is common in Egyptian dialectal writing.
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)

    # Remove non-essential special characters but PRESERVE Latin for code-switching - 
    # Keeps Arabic, English, Numbers, and basic punctuation
    text = re.sub(r'[^\u0600-\u06FF a-zA-Z0-9\s.,!؟]', '', text)
    
    # Collapse multiple whitespaces
    text = " ".join(text.split())
    
    return text

# 3. PROCESS TRANSCRIPTS (.txt) - [cite: 49, 50]
print("--- CLEANING TRANSCRIPTS ---")
txt_files = [f for f in os.listdir(raw_path) if f.endswith('.txt')]

for f_name in txt_files:
    with open(os.path.join(raw_path, f_name), "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove timestamp markers like [00:15] or 12.205: - 
    clean_txt = re.sub(r'(\[\d+:\d+\]|\d+(\.\d+)?:\s*)', '', content)
    
    final_txt = normalize_arabic(clean_txt)
    
    with open(os.path.join(txt_out, f"cleaned_{f_name}"), "w", encoding="utf-8") as f:
        f.write(final_txt)
    print(f"Done Processed: {f_name}")

# 4. PROCESS QA PAIRS (.csv) - [cite: 54, 55, 56]
print("\n--- CLEANING QA PAIRS ---")
csv_files = [f for f in os.listdir(raw_path) if f.endswith('.csv')]

for f_name in csv_files:
    df = pd.read_csv(os.path.join(raw_path, f_name))
    
    # Clean both questions and answers - [cite: 67]
    df['question'] = df['question'].apply(normalize_arabic)
    df['answer'] = df['answer'].apply(normalize_arabic)
    
    df.to_csv(os.path.join(csv_out, f"cleaned_{f_name}"), index=False, encoding='utf-8-sig')
    print(f"Done Processed: {f_name}")

# 5. ANALYSIS & STATISTICS (For your 2-page Report) - [cite: 16, 64, 65]
print("\n--- GENERATING DATASET INSIGHTS ---")

all_text = ""
for f_name in os.listdir(txt_out):
    with open(os.path.join(txt_out, f_name), "r", encoding="utf-8") as f:
        all_text += f.read() + " "

tokens = all_text.split()
total_words = len(tokens)
vocab_size = len(set(tokens))

# Identify Latin tokens for code-switching analysis - 
latin_tokens = [t for t in tokens if re.search(r'[a-zA-Z]', t)]
code_switch_pct = (len(latin_tokens) / total_words) * 100

print(f"Total Transcript Words: {total_words}")
print(f"Unique Vocabulary: {vocab_size}")
print(f"Code-Switching (English) Percentage: {code_switch_pct:.2f}%")

# Traceability Check: Ensure answers are still spans of transcripts - [cite: 57, 58]
print("\n--- TRACEABILITY CHECK ---")
sample_csv = pd.read_csv(os.path.join(csv_out, f"cleaned_{csv_files[0]}"))
with open(os.path.join(txt_out, f"cleaned_{txt_files[0]}"), "r", encoding="utf-8") as f:
    sample_transcript = f.read()

found_count = 0
for ans in sample_csv['answer'].head(20):
    if str(ans) in sample_transcript:
        found_count += 1
print(f"Traceability Rate (Sample of 20): {(found_count/20)*100}%")

# 6. PREPARE FOR MILESTONE 2 (Token Index) - 
word_to_idx = {word: i for i, word in enumerate(sorted(list(set(tokens))))}
with open("token_index.json", "w", encoding="utf-8") as f:
    json.dump(word_to_idx, f, ensure_ascii=False, indent=4)

print("\n ALL STEPS COMPLETE.")