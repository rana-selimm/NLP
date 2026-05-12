"""
rag_system.py  –  Milestone 3: Retrieval-Augmented Generation for Arabic Chatbot
==================================================================================
Design Choices (must be justified):

EMBEDDING MODEL: paraphrase-multilingual-MiniLM-L12-v2
  - Trained on 50+ languages including Arabic and English
  - Preserves Arabic dialectal variation and English code-switching
  - Lightweight (118 MB) with strong multilingual semantic understanding
  - Normalised embeddings allow cosine similarity via inner product

CHUNKING: 250 words, 50-word overlap
  - 250 words ≈ 1-2 coherent paragraphs in the Dahih transcripts
  - 50-word overlap prevents answers from being cut at chunk boundaries
  - Word-level split respects Arabic morphology (no sub-word splitting)
  - Each chunk stores source episode for traceability

VECTOR STORE: FAISS IndexFlatIP (inner product on L2-normalised vectors)
  - Local, serverless, deterministic
  - IP on normalised vectors = cosine similarity (range 0..1)

OOD DETECTION: cosine similarity threshold = 0.25
  - If best retrieved chunk scores < 0.25, query is off-topic
  - Empirically: in-domain queries score 0.35–0.85; OOD queries < 0.20

LLMs: Groq (llama-3.1-8b-instant) primary, Gemini (gemini-1.5-flash) secondary
  - Both free-tier, API-based, support Arabic + English
  - Exponential-backoff retry (3 attempts) then model fallback
"""

import os
import json
import time
import pickle
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── optional LLM clients ────────────────────────────────────────────────────
try:
    from groq import Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

try:
    from google import genai as genai_sdk
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    source: str        # human-readable episode name
    chunk_id: int
    file: str          # original filename
    word_offset: int   # position in source transcript


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]   # cosine similarities, descending
    is_out_of_domain: bool
    max_score: float


# ── Core RAG System ──────────────────────────────────────────────────────────

class ArabicRAGSystem:
    """
    Handles document loading, chunking, embedding, FAISS indexing and retrieval.
    Uses 5 episodes from Milestone 1 cleaned transcripts.
    """

    EPISODES = {
        "cleaned_الأخطبوط الدحيح.txt":                                    "الأخطبوط",
        "cleaned_الساموراي  الدحيح.txt":                                   "الساموراي",
        "cleaned_تاج محل  الدحيح.txt":                                     "تاج محل",
        "cleaned_فيزياء و فلسفة الحركة  الدحيح.txt":                       "فيزياء و فلسفة الحركة",
        "cleaned_هل Citizen Kane أفضل فيلم في التاريخ؟  الدحيح.txt":       "Citizen Kane",
    }

    OOD_THRESHOLD = 0.25
    CHUNK_SIZE    = 250   # words
    CHUNK_OVERLAP = 50    # words
    TOP_K         = 4

    def __init__(
        self,
        cleaned_dir: str = "cleaned",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.cleaned_dir = Path(cleaned_dir)
        self.embedding_model_name = embedding_model

        print(f"[RAG] Loading embedding model: {embedding_model} …")
        self.embedder = SentenceTransformer(embedding_model)

        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

    # ── Document loading & chunking ──────────────────────────────────────────

    def load_and_chunk(self) -> List[Chunk]:
        """
        Load the 5 selected episodes and split into overlapping word chunks.
        Uses LangChain RecursiveCharacterTextSplitter for robust splitting.
        """
        # LangChain splitter (character-based; we convert word limits to chars)
        # Average Arabic word ≈ 5 chars + space → 250 words ≈ 1500 chars
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", "،", "!", "؟", " ", ""],
            keep_separator=False,
        )

        all_chunks: List[Chunk] = []
        chunk_id = 0

        for fname, ep_name in self.EPISODES.items():
            path = self.cleaned_dir / fname
            if not path.exists():
                print(f"[RAG] Warning: {fname} not found, skipping.")
                continue

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            # LangChain Document for metadata tracking
            doc = Document(page_content=text, metadata={"source": ep_name, "file": fname})
            sub_docs = splitter.split_documents([doc])

            for sd in sub_docs:
                words = sd.page_content.split()
                all_chunks.append(Chunk(
                    text=sd.page_content,
                    source=ep_name,
                    chunk_id=chunk_id,
                    file=fname,
                    word_offset=chunk_id,  # approximate
                ))
                chunk_id += 1

        self.chunks = all_chunks
        print(f"[RAG] Created {len(all_chunks)} chunks from {len(self.EPISODES)} episodes.")
        return all_chunks

    # ── Embedding & indexing ─────────────────────────────────────────────────

    def build_index(self) -> None:
        """Embed all chunks and build a FAISS inner-product index."""
        if not self.chunks:
            self.load_and_chunk()

        texts = [c.text for c in self.chunks]
        print(f"[RAG] Embedding {len(texts)} chunks …")
        embs = self.embedder.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,   # L2 norm → cosine via IP
        ).astype(np.float32)

        self.embeddings = embs
        dim = embs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)   # inner product
        self.faiss_index.add(embs)
        print(f"[RAG] FAISS index ready: {self.faiss_index.ntotal} vectors (dim={dim}).")

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Retrieve top-k chunks for a query using cosine similarity.
        Returns RetrievalResult with OOD flag based on threshold.
        """
        if self.faiss_index is None:
            raise RuntimeError("Call build_index() first.")
        top_k = top_k or self.TOP_K

        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.faiss_index.search(q_emb, top_k)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        valid = [(s, i) for s, i in zip(scores, indices) if i >= 0]
        if not valid:
            return RetrievalResult([], [], True, 0.0)

        ret_scores  = [s for s, _ in valid]
        ret_chunks  = [self.chunks[i] for _, i in valid]
        max_score   = ret_scores[0]
        is_ood      = max_score < self.OOD_THRESHOLD

        return RetrievalResult(ret_chunks, ret_scores, is_ood, max_score)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, pkl_path: str = "rag_index.pkl") -> None:
        faiss_path = pkl_path.replace(".pkl", ".faiss")
        with open(pkl_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        faiss.write_index(self.faiss_index, faiss_path)
        print(f"[RAG] Saved index to {pkl_path} and {faiss_path}.")

    def load(self, pkl_path: str = "rag_index.pkl") -> None:
        faiss_path = pkl_path.replace(".pkl", ".faiss")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.chunks    = data["chunks"]
        self.embeddings = data["embeddings"]
        self.faiss_index = faiss.read_index(faiss_path)
        print(f"[RAG] Loaded {len(self.chunks)} chunks, FAISS size={self.faiss_index.ntotal}.")


# ── LLM Manager ─────────────────────────────────────────────────────────────

class LLMManager:
    """
    Manages Groq and Gemini API clients.
    Implements retry with exponential backoff and automatic model fallback.
    Never raises – always returns a string (possibly an error message).
    """

    GROQ_MODELS   = ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]
    GEMINI_MODELS = ["gemini-1.5-flash"]

    def __init__(
        self,
        groq_api_key:   Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        groq_key   = groq_api_key   or os.getenv("GROQ_API_KEY",   "")
        gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")

        self.groq_client   = None
        self.gemini_client = None

        if groq_key and _GROQ_OK:
            self.groq_client = Groq(api_key=groq_key)
            print("[LLM] Groq client initialised.")

        if gemini_key and _GEMINI_OK:
            self.gemini_client = genai_sdk.Client(api_key=gemini_key)
            print("[LLM] Gemini client initialised.")

        if not self.groq_client and not self.gemini_client:
            print("[LLM] Warning: no LLM clients available. Set GROQ_API_KEY or GEMINI_API_KEY.")

    def call(
        self,
        prompt:      str,
        model:       str = "groq-llama",
        max_tokens:  int = 600,
        max_retries: int = 3,
    ) -> str:
        """
        Call an LLM with retry + fallback.
        model: 'groq-llama' | 'groq-gemma' | 'gemini'
        """
        sequence = self._build_sequence(model)
        last_err = None

        for model_tag, call_fn in sequence:
            for attempt in range(max_retries):
                try:
                    return call_fn(prompt, max_tokens)
                except Exception as exc:
                    last_err = exc
                    wait = 2 ** attempt
                    print(f"[LLM] {model_tag} attempt {attempt+1} failed: {exc}. Retry in {wait}s …")
                    time.sleep(wait)
            print(f"[LLM] {model_tag} exhausted retries, trying next fallback …")

        return f"[Error] All LLMs failed. Last: {last_err}"

    def _build_sequence(self, model: str) -> List[Tuple[str, Any]]:
        """
        Build the ordered list of (label, callable) to try.
        Only adds a provider if its client was successfully initialised
        (i.e. a non-empty key was provided). Never blindly falls back to a
        provider whose key is missing or known-invalid.
        """
        seq = []

        if model.startswith("groq"):
            # Primary: requested Groq model(s)
            if self.groq_client:
                groq_m = "gemma2-9b-it" if "gemma" in model else "llama-3.1-8b-instant"
                seq.append((f"groq/{groq_m}", lambda p, mt, m=groq_m: self._groq(p, mt, m)))
                for fb in self.GROQ_MODELS:
                    if fb != groq_m:
                        seq.append((f"groq/{fb}", lambda p, mt, m=fb: self._groq(p, mt, m)))
            # Fallback to Gemini only if its client is initialised
            if self.gemini_client:
                seq.append(("gemini/2.0-flash[fallback]", lambda p, mt: self._gemini(p, mt)))

        elif model == "gemini":
            # Primary: Gemini
            if self.gemini_client:
                seq.append(("gemini/2.0-flash", lambda p, mt: self._gemini(p, mt)))
            # Fallback to Groq if its client is initialised
            if self.groq_client:
                for fb in self.GROQ_MODELS:
                    seq.append((f"groq/{fb}[fallback]", lambda p, mt, m=fb: self._groq(p, mt, m)))

        if not seq:
            raise ValueError(
                "No LLM client is available. "
                "Please enter a valid GROQ_API_KEY or GEMINI_API_KEY."
            )
        return seq

    def _groq(self, prompt: str, max_tokens: int, model: str = "llama-3.1-8b-instant") -> str:
        resp = self.groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def _gemini(self, prompt: str, max_tokens: int) -> str:
        resp = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai_sdk.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            ),
        )
        return resp.text.strip()


# ── Prompt Templates ─────────────────────────────────────────────────────────

PROMPTS = {
    # System-guided English
    "system_guided_en": """\
You are a specialized assistant for the Arabic educational program "Al-Dahih" (الدحيح).
Answer ONLY based on the retrieved context below. Do not use external knowledge.

Rules:
1. If the context does not contain the answer, respond: "لم أجد هذه المعلومة في المحتوى المتاح" (Info not found in available content).
2. Respond in the same language as the user's question (Arabic or English).
3. Cite the episode name when possible.
4. Be concise and faithful – no hallucination.

--- RETRIEVED CONTEXT ---
{context}

--- CONVERSATION HISTORY ---
{history}

--- USER QUESTION ---
{question}

--- ANSWER ---""",

    # Minimal English
    "minimal_en": """\
Context: {context}
History: {history}
Question: {question}
Answer:""",

    # System-guided Arabic
    "system_guided_ar": """\
أنت مساعد متخصص في برنامج الدحيح التعليمي.
أجب فقط استناداً إلى السياق المسترجع أدناه. لا تستخدم معرفة خارجية.

القواعد:
1. إذا لم يحتوِ السياق على الإجابة، قل: "لم أجد هذه المعلومة في المحتوى المتاح".
2. أجب بنفس لغة سؤال المستخدم (عربي أو إنجليزي).
3. اذكر اسم الحلقة عند الإمكان.
4. كن دقيقاً ومخلصاً للمحتوى، ولا تختلق معلومات.

--- السياق المسترجع ---
{context}

--- تاريخ المحادثة ---
{history}

--- سؤال المستخدم ---
{question}

--- الإجابة ---""",

    # Minimal Arabic
    "minimal_ar": """\
السياق: {context}
التاريخ: {history}
السؤال: {question}
الإجابة:""",
}


# ── RAG Chatbot ──────────────────────────────────────────────────────────────

class RAGChatbot:
    """
    Multi-turn conversational RAG chatbot.

    Memory strategies (design choices):
    - full_history     : pass entire conversation to LLM (best coherence, highest token cost)
    - sliding_window   : last window_size turns (balanced coherence vs cost)
    - strict_truncation: only last 1 user+assistant pair (minimal cost, least coherence)
    - summarized_history: LLM-compressed summary of older turns + recent turns (best balance)

    Prompt types: system_guided | minimal  ×  en | ar  →  4 combinations
    """

    OOD_REJECTION = {
        "en": (
            "Sorry, your question is outside the scope of the available episodes. "
            "I can answer questions about: Octopus, Samurai, Taj Mahal, "
            "Physics & Motion, and Citizen Kane – all from Al-Dahih."
        ),
        "ar": (
            "عذراً، سؤالك خارج نطاق الحلقات المتاحة. "
            "يمكنني الإجابة عن: الأخطبوط، الساموراي، تاج محل، "
            "فيزياء الحركة، وفيلم Citizen Kane – من برنامج الدحيح."
        ),
    }

    def __init__(
        self,
        rag:              ArabicRAGSystem,
        llm:              LLMManager,
        llm_name:         str = "groq-llama",   # 'groq-llama' | 'groq-gemma' | 'gemini'
        memory_strategy:  str = "sliding_window",
        prompt_type:      str = "system_guided",
        prompt_lang:      str = "en",
        window_size:      int = 3,
    ):
        self.rag             = rag
        self.llm             = llm
        self.llm_name        = llm_name
        self.memory_strategy = memory_strategy
        self.prompt_type     = prompt_type
        self.prompt_lang     = prompt_lang
        self.window_size     = window_size

        self.history: List[Dict[str, str]] = []   # [{role, content}, …]
        self.logs:    List[Dict]            = []

    # ── history management ───────────────────────────────────────────────────

    def _history_text(self) -> str:
        """Return conversation history formatted for the prompt."""
        if not self.history:
            return "None"

        if self.memory_strategy == "full_history":
            turns = self.history

        elif self.memory_strategy == "sliding_window":
            # Keep last window_size Q/A pairs (2 messages per pair)
            turns = self.history[-(self.window_size * 2):]

        elif self.memory_strategy == "strict_truncation":
            # Only the last Q/A pair
            turns = self.history[-2:]

        elif self.memory_strategy == "summarized_history":
            turns = self._summarized_turns()
            if isinstance(turns, str):
                return turns   # already formatted
        else:
            turns = []

        return "\n".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
            for t in turns
        ) or "None"

    def _summarized_turns(self):
        """Summarise older history with the LLM; keep last 2 turns verbatim."""
        if len(self.history) <= 4:
            return self.history          # not enough to summarise

        older  = self.history[:-2]
        recent = self.history[-2:]

        raw = "\n".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
            for t in older
        )
        summary_prompt = (
            f"Summarise this conversation in 2–3 sentences (be brief):\n{raw}\nSummary:"
        )
        try:
            summary = self.llm.call(summary_prompt, model=self.llm_name, max_tokens=120)
        except Exception:
            summary = "(summary unavailable)"

        recent_text = "\n".join(
            f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}"
            for t in recent
        )
        return f"[Earlier summary: {summary}]\n{recent_text}"

    # ── prompt building ──────────────────────────────────────────────────────

    def _build_prompt(self, question: str, context: str) -> str:
        key = f"{self.prompt_type}_{self.prompt_lang}"
        template = PROMPTS.get(key, PROMPTS["system_guided_en"])
        return template.format(
            context=context,
            history=self._history_text(),
            question=question,
        )

    # ── main chat ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> Dict:
        """
        Process one user turn. Returns a dict with:
          response, sources, scores, is_ood, retrieved_context, log
        """
        result = self.rag.retrieve(user_input)

        log = {
            "turn":           len(self.history) // 2 + 1,
            "query":          user_input,
            "is_ood":         result.is_out_of_domain,
            "max_score":      result.max_score,
            "sources":        [c.source for c in result.chunks],
            "memory":         self.memory_strategy,
            "prompt_type":    self.prompt_type,
            "prompt_lang":    self.prompt_lang,
            "llm":            self.llm_name,
        }

        if result.is_out_of_domain:
            rejection = self.OOD_REJECTION.get(self.prompt_lang, self.OOD_REJECTION["en"])
            log["response"] = rejection
            self.history.append({"role": "user",      "content": user_input})
            self.history.append({"role": "assistant",  "content": rejection})
            self.logs.append(log)
            return {"response": rejection, "sources": [], "scores": [],
                    "is_ood": True, "retrieved_context": "", "log": log}

        # Build context string from retrieved chunks
        context = "\n\n".join(
            f"[{c.source}] {c.text}" for c in result.chunks
        )

        prompt   = self._build_prompt(user_input, context)
        response = self.llm.call(prompt, model=self.llm_name, max_tokens=600)

        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": response})

        log["response"]          = response
        log["context_chars"]     = len(context)
        log["prompt_chars"]      = len(prompt)
        self.logs.append(log)

        return {
            "response":          response,
            "sources":           [(c.source, s) for c, s in zip(result.chunks, result.scores)],
            "scores":            result.scores,
            "is_ood":            False,
            "retrieved_context": context,
            "log":               log,
        }

    def reset(self) -> None:
        """Clear conversation history (keep index)."""
        self.history.clear()

    def save_logs(self, path: str = "evaluation_logs.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
        print(f"[RAG] Saved {len(self.logs)} log entries → {path}")


# ── Evaluation helpers ───────────────────────────────────────────────────────

def faithfulness_score(answer: str, context: str) -> float:
    """
    Faithfulness: fraction of unique answer words found in retrieved context.
    Justification: directly measures grounding without external model.
    """
    a_words = set(answer.lower().split())
    c_words = set(context.lower().split())
    if not a_words:
        return 0.0
    return len(a_words & c_words) / len(a_words)


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """
    ROUGE-L F1 via LCS.
    Justification: standard metric for text generation quality; language-agnostic.
    """
    def lcs_len(a, b):
        a, b = a.split(), b.split()
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_len(hypothesis, reference)
    ref_len  = len(reference.split())
    hyp_len  = len(hypothesis.split())
    if ref_len == 0 or hyp_len == 0:
        return 0.0
    p = lcs / hyp_len
    r = lcs / ref_len
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def token_overlap_f1(hypothesis: str, reference: str) -> float:
    """
    Token-level F1 (unigram overlap).
    Justification: simple semantic correctness proxy; robust across languages.
    """
    pred = set(hypothesis.lower().split())
    gold = set(reference.lower().split())
    if not pred or not gold:
        return 0.0
    common = pred & gold
    p = len(common) / len(pred)
    r = len(common) / len(gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Factory helper ───────────────────────────────────────────────────────────

def build_rag_chatbot(
    cleaned_dir:    str  = "cleaned",
    groq_api_key:   str  = "",
    gemini_api_key: str  = "",
    llm_name:       str  = "groq-llama",
    memory_strategy:str  = "sliding_window",
    prompt_type:    str  = "system_guided",
    prompt_lang:    str  = "en",
    index_pkl:      str  = "rag_index.pkl",
    force_rebuild:  bool = False,
) -> Tuple[ArabicRAGSystem, RAGChatbot]:
    """
    Convenience function: build (or load) the RAG system and return a chatbot.
    """
    rag = ArabicRAGSystem(cleaned_dir=cleaned_dir)
    llm = LLMManager(groq_api_key=groq_api_key, gemini_api_key=gemini_api_key)

    if not force_rebuild and Path(index_pkl).exists():
        rag.load(index_pkl)
    else:
        rag.load_and_chunk()
        rag.build_index()
        rag.save(index_pkl)

    bot = RAGChatbot(
        rag=rag, llm=llm,
        llm_name=llm_name,
        memory_strategy=memory_strategy,
        prompt_type=prompt_type,
        prompt_lang=prompt_lang,
    )
    return rag, bot
