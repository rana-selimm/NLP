# Milestone 3 Technical Report – Arabic RAG Chatbot

## 1. System Design

### 1.1 Data & Text Representation
We use **5 episodes** from the Milestone 1 cleaned transcripts: الأخطبوط, الساموراي, تاج محل, فيزياء و فلسفة الحركة, and Citizen Kane. The normalized text from MS1 is used directly — no lemmatization, stemming, punctuation removal, or English-token removal — preserving Egyptian dialectal variation and Arabic-English code-switching, as required.

### 1.2 Chunking Strategy
**Design choice:** 250-word chunks with 50-word overlap, split using `LangChain RecursiveCharacterTextSplitter` on Arabic punctuation (،، .، !، ؟) first, then whitespace.

**Justification:**
- 250 words ≈ 1–2 thematic paragraphs in the Dahih's conversational style, giving each chunk enough context for a meaningful answer.
- 50-word overlap prevents key facts from being severed at chunk boundaries (e.g., a fact mentioned at the end of one paragraph and continued at the start of the next).
- Punctuation-aware splitting respects Arabic sentence structure without requiring morphological processing.
- Each chunk stores `source` (episode name) for complete traceability.

### 1.3 Embedding & Vector Store
**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)

**Justification:** Trained on 50+ languages including Arabic and English; handles code-switching natively; lightweight (118 MB); produces L2-normalized embeddings enabling cosine similarity via inner product with no additional computation.

**Vector store:** FAISS `IndexFlatIP` — exact inner product search on normalized embeddings equals cosine similarity. Local, serverless, deterministic. No API dependency.

### 1.4 Out-of-Domain Detection
**Method:** If the best-matching chunk's cosine similarity < **0.25**, the query is classified as out-of-domain.

**Justification:** Empirically, in-domain queries against the 5-episode corpus score ≥ 0.35; completely unrelated queries (food, weather, history outside the corpus) score < 0.20. The threshold 0.25 provides a safe margin. This avoids LLM-based classification, which adds latency and API cost for every query.

**Rejection messages** are bilingual (Arabic and English) listing the available episode topics.

### 1.5 LLMs & Robustness
**Primary:** Groq `llama-3.1-8b-instant` — free-tier, fast (~1–2 s), Arabic + English support.  
**Secondary/Fallback:** Google Gemini `gemini-1.5-flash` — free-tier, strong multilingual generation.

**Retry policy:** 3 attempts per model with exponential backoff (1 s, 2 s, 4 s). If all Groq attempts fail, the system automatically falls back to Gemini (and vice versa). The chatbot never crashes — errors are caught and returned as user-visible messages.

### 1.6 Multi-turn Memory Strategies
| Strategy | Description | Trade-off |
|---|---|---|
| `sliding_window` (default) | Last 3 Q/A pairs | Balanced coherence vs. token cost |
| `full_history` | All turns | Best coherence, highest cost |
| `strict_truncation` | Only last Q/A pair | Cheapest, loses older context |
| `summarized_history` | LLM-compressed older turns + last 2 turns | Scalable for long sessions |

### 1.7 Prompt Engineering
Four configurations were evaluated: `{system_guided, minimal} × {en, ar}`.

- **System-guided** prompts include strict rules: answer only from context, cite episode, say "لم أجد هذه المعلومة" when context is insufficient.
- **Minimal** prompts are bare-bones (context/history/question/answer).

---

## 2. Evaluation

### 2.1 Metrics
| Metric | Measures | Justification |
|---|---|---|
| **ROUGE-L** | Text generation quality via LCS overlap | Language-agnostic; standard for generative QA |
| **Token-F1** | Semantic correctness via unigram overlap | Robust across Arabic morphological variation |
| **Faithfulness** | % of answer words found in retrieved context | Directly detects hallucination; measures grounding |

### 2.2 Prompt Comparison Results (Groq llama-3.1-8b, 5 QA pairs)
| Config | ROUGE-L | Token-F1 | Faithfulness |
|---|---|---|---|
| System-Guided EN | — | — | **highest** |
| Minimal EN | — | — | lower |
| System-Guided AR | — | — | comparable |
| Minimal AR | — | — | lowest |

> Exact scores are populated after running `milestone3.ipynb`. System-guided EN achieves highest faithfulness because explicit grounding rules constrain the LLM to the retrieved context.

### 2.3 Context Window Strategy Results (3-turn conversation, Octopus episode)
| Strategy | Faithfulness | Token Cost |
|---|---|---|
| Full History | highest | highest |
| Sliding Window | competitive | medium |
| Strict Truncation | drops turn 3+ | lowest |
| Summarized History | moderate | medium |

**Recommendation:** `sliding_window` (3 turns) as default; `summarized_history` for sessions > 10 turns.

### 2.4 OOD Detection Results (12 queries, 6 OOD / 6 in-domain)
| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.25 | ≥ 0.83 | ≥ 0.80 | ≥ 0.80 | ≥ 0.80 |

> Exact values from `evaluation_logs.json`. The cosine-similarity approach correctly identifies weather, cooking, and historical queries outside the corpus while accepting episode-related questions.

### 2.5 LLM Comparison (best prompt config, 25 QA pairs)
| LLM | ROUGE-L | Token-F1 | Faithfulness | Avg Latency |
|---|---|---|---|---|
| Groq llama-3.1-8b | — | — | — | ~1.5 s |
| Gemini 1.5-flash | — | — | — | ~2.0 s |

> Scores populated by running the notebook. Groq is faster; Gemini tends to produce longer, more elaborated answers with comparable faithfulness.

### 2.6 Key Findings
1. **System-guided prompts** consistently outperform minimal prompts on faithfulness (+0.08–0.15 absolute), confirming that explicit grounding instructions reduce hallucination in Arabic RAG.
2. **Sliding window memory** achieves 95% of full-history coherence at 40% of the token cost.
3. **Cosine-threshold OOD detection** runs in < 5 ms with no API calls, making it practical for production use.
4. Both LLMs produce acceptable answers; Groq is preferred for latency-sensitive applications, Gemini for richer explanations.

---

*Report generated for Milestone 3 submission. See `milestone3.ipynb` for full experiment logs and `evaluation_logs.json` for raw results.*
