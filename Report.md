Technical Report: Milestone 1 - Data Preprocessing & Analysis

1. Dataset Understanding
Structure
The dataset is organized into a dual-folder structure containing the raw data for "ElDatee7" Season 8:

Transcripts: 13 .txt files containing the full spoken content.

QA Pairs: 13 .csv files containing supervised question-answer pairs (approx. 300 per video).

Dataset Statistics (from EDA)
Total QA Pairs: 3,890

Avg Question Length: 6.08 words

Avg Answer Length: 4.93 words

Final Vocabulary Size: 24,165 tokens (including special markers)

Linguistic Characteristics
The dataset is a complex mixture of:

Modern Standard Arabic (MSA): Used for formal scientific facts.

Egyptian Dialect: The host’s natural conversational style.

English Code-Switching: Technical terms (e.g., DNA, enzymes) written in Latin characters.


2. Text Exploration / EDA
Analysis of Patterns
Our EDA found that the dataset is highly conversational. We detected a 1.16% Code-Switching rate, highlighting the presence of foreign technical tokens.

Frequency Analysis
The word frequency analysis showed that dialectal markers are the most common tokens:

"اللي" (1,689 occurrences)

"مش" (571 occurrences)


3. Noise Detection and Data Quality
Identified Noise
We identified several "noise" factors that prevent efficient modeling:

Timestamps: Formatting like [00:15] or 12.205: which are non-semantic.

Orthographic Inconsistencies: Variability in Arabic letters (e.g., أ vs ا).

Punctuation Noise: Repeated symbols like ؟!!!.

Examples of Noisy Data
Raw: [00:45] الدحيح بيقول: إنّ القهوة دي مسألة حياة أو موت.. مش كدة؟!!!

Cleaned: الدحيح بيقول ان القهوه دي مساله حياه او موت مش كده ؟

Why Cleaning is Necessary
Cleaning reduces Vocabulary Sparsity. Without normalization, the model treats أنا and انا as two different words, which splits the training data and weakens the model's ability to learn word meanings.


4. Arabic Text Normalization
Approach
We implemented a normalize_arabic function to:

Standardize Alif: Map أ، إ، آ to a plain ا.

Standardize Suffixes: Map ى to ي and ة to ه.

Tashkeel Removal: Strip all diacritics to simplify the vocabulary.

Impact
This normalization ensures that dialectal variations and spelling errors are collapsed into a single representation, significantly lowering the "Unique Vocabulary" count and improving the signal-to-noise ratio.


5. Tokenization & Text Representation
Chosen Approach
We utilized Whitespace Tokenization. This approach is ideal for the initial milestone to establish a clear word-to-index mapping.

Vocabulary Indexing
We built a token_index.json file that maps every word to a unique integer.

Handling Rare Tokens
We explicitly added a <UNK> token to handle out-of-vocabulary (OOV) words. We also included <PAD>, <SOS>, and <EOS> tokens, which are essential requirements for the neural architectures in Milestone 2.



6. Data Preparation for Modeling
Traceability Check
A final check confirmed that after cleaning, the answers in our CSVs still match their locations in the transcripts. This ensures the data is ready for the Retrieval-Augmented Generation (RAG) tasks.

Limitations & Future Solutions
Limitation: Simple whitespace tokenization ignores the complex morphology of Arabic (where one word can contain a preposition, a verb, and a pronoun).

Future Solution: For Milestone 2, we plan to implement Subword Tokenization (BPE). This will allow the neural model to learn sub-units of words, better handling the Egyptian dialect's unique prefixes and suffixes.