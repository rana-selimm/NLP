import os
import re
import pandas as pd

# 1. SETUP PATHS
# We define where things are at the top so it's easy to change later
raw_path = "data"
txt_out = "cleaned"
csv_out = "cleaned_qa"

# Create folders if they don't exist
for folder in [txt_out, csv_out]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 2. THE CLEANING "MACHINE"
def normalize_arabic(text):
    """Standardizes Arabic characters to reduce noise for the AI model."""
    if not isinstance(text, str) or text == 'nan': 
        return ""
    
    # Normalize Alif shapes (أ، إ، آ -> ا)
    text = re.sub("[إأآ]", "ا", text)
    # Normalize final Ya and Ta Marbuta (ى -> ي, ة -> ه)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    # Remove Tashkeel (vowels/diacritics)
    tashkeel = re.compile(r'[\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    return text

# 3. PROCESS TRANSCRIPTS (.txt)
print("--- STARTING TRANSCRIPT CLEANING ---")
txt_files = [f for f in os.listdir(raw_path) if f.endswith('.txt')]

for f_name in txt_files:
    with open(os.path.join(raw_path, f_name), "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove timestamps (e.g., 0.0: or 12.205:)
    clean_txt = re.sub(r'\d+(\.\d+)?:\s*', '', content)
    # Normalize Arabic
    final_txt = normalize_arabic(clean_txt)
    
    # Save the cleaned file
    with open(os.path.join(txt_out, f"cleaned_{f_name}"), "w", encoding="utf-8") as f:
        f.write(final_txt)
    print(f"✔ Processed: {f_name}")

# 4. PROCESS QA PAIRS (.csv)
print("\n--- STARTING CSV CLEANING ---")
csv_files = [f for f in os.listdir(raw_path) if f.endswith('.csv')]

for f_name in csv_files:
    # Read the CSV
    df = pd.read_csv(os.path.join(raw_path, f_name))
    
    # Apply normalization to the 'question' and 'answer' columns
    df['question'] = df['question'].apply(normalize_arabic)
    df['answer'] = df['answer'].apply(normalize_arabic)
    
    # Save the cleaned CSV (encoding 'utf-8-sig' ensures Arabic shows correctly in Excel)
    df.to_csv(os.path.join(csv_out, f"cleaned_{f_name}"), index=False, encoding='utf-8-sig')
    print(f"✔ Processed: {f_name}")

print("\n🎉 ALL DATA CLEANED, NORMALIZED, AND SAVED!")

# --- MASTER STATS & ANALYSIS (Covers Checklist #1, #2, #3, & #5) ---
print("\n--- GENERATING FINAL CHECKLIST STATISTICS ---")

# 1. Aggregating Text for Analysis
all_text = ""
for f_name in os.listdir(txt_out):
    with open(os.path.join(txt_out, f_name), "r", encoding="utf-8") as f:
        all_text += f.read() + " "

# 2. Dataset Statistics (Checklist #1 & #2)
tokens = all_text.split()
total_word_count = len(tokens)

# Load all cleaned CSVs to calculate QA statistics
all_qa_files = [pd.read_csv(os.path.join(csv_out, f)) for f in os.listdir(csv_out)]
full_df = pd.concat(all_qa_files)

# Ensure columns are treated as strings for length calculation
full_df['q_len'] = full_df['question'].astype(str).str.split().str.len()
full_df['a_len'] = full_df['answer'].astype(str).str.split().str.len()

print(f"Total Word Count in Transcripts: {total_word_count}")
print(f"Total QA Pairs: {len(full_df)}")
print(f"Average Question Length: {full_df['q_len'].mean():.2f} words")
print(f"Average Answer Length: {full_df['a_len'].mean():.2f} words")

# 3. Linguistic Evidence: Code-Switching & Dialect (Checklist #1 & #3)
latin_pattern = re.compile(r'[a-zA-Z]+')
latin_tokens = [t for t in tokens if latin_pattern.search(t)]
code_switch_pct = (len(latin_tokens) / total_word_count) * 100

dialect_markers = ["مش", "اللي", "كدا", "دا", "دي", "بتاع"]
marker_counts = {word: tokens.count(word) for word in dialect_markers}

print(f"\nVocabulary Size: {len(set(tokens))}")
print(f"English/Latin tokens: {len(latin_tokens)} ({code_switch_pct:.2f}% of data)")
print(f"Example English terms: {list(set(latin_tokens))[:10]}")

print("\nDialect Marker Frequency:")
for word, count in marker_counts.items():
    print(f"- {word}: {count} occurrences")

# 4. Tokenization & Token Index (Checklist #5)
# Building a mapping of word -> unique integer
vocab = sorted(list(set(tokens)))
word_to_idx = {word: i for i, word in enumerate(vocab)}

# 5. Saving Artifacts for Milestone 2
# Save the Vocabulary List
with open("vocab.txt", "w", encoding="utf-8") as f:
    for word in vocab:
        f.write(word + "\n")

# Save the Token Index as a JSON file (Requirement for neural architectures)
import json
with open("token_index.json", "w", encoding="utf-8") as f:
    json.dump(word_to_idx, f, ensure_ascii=False, indent=4)

print("\n🎉 ALL CHECKLIST REQUIREMENTS GENERATED AND SAVED!")

def normalize_arabic(text):
    if not isinstance(text, str) or text == 'nan': 
        return ""
    
    # NEW: Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text) 
    
    text = re.sub("[إأآ]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    tashkeel = re.compile(r'[\u064B-\u0652]')
    text = re.sub(tashkeel, '', text)
    return text