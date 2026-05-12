"""
app.py – Streamlit Multi-turn RAG Chatbot Interface (Milestone 3)
=================================================================
Run:  streamlit run app.py
"""

import os, json, time
from pathlib import Path

import streamlit as st

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic RAG Chatbot – الدحيح",
    page_icon="🎓",
    layout="wide",
)

# ── custom CSS for RTL Arabic text ───────────────────────────────────────────
st.markdown("""
<style>
.rtl { direction: rtl; text-align: right; font-family: 'Segoe UI', Tahoma, sans-serif; }
.source-badge {
    background: #e8f4fd; border-radius: 6px; padding: 2px 8px;
    font-size: 0.78em; margin: 2px; display: inline-block; color: #1a6fa0;
}
.score-badge {
    background: #f0f9e8; border-radius: 6px; padding: 2px 8px;
    font-size: 0.78em; color: #2e7d32;
}
.ood-msg { color: #b71c1c; font-weight: 600; }
.log-entry { font-size: 0.8em; color: #555; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# ── lazy import of RAG system ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG system and building index…")
def load_rag(groq_key: str, gemini_key: str, force_rebuild: bool = False):
    """Build / reload the RAG system (cached across Streamlit reruns)."""
    from rag_system import build_rag_chatbot
    rag, _ = build_rag_chatbot(
        cleaned_dir    = "cleaned",
        groq_api_key   = groq_key,
        gemini_api_key = gemini_key,
        force_rebuild  = force_rebuild,
    )
    return rag


def make_bot(rag, groq_key, gemini_key, llm_name, memory, ptype, plang):
    from rag_system import LLMManager, RAGChatbot
    llm = LLMManager(groq_api_key=groq_key, gemini_api_key=gemini_key)
    return RAGChatbot(
        rag=rag, llm=llm,
        llm_name=llm_name,
        memory_strategy=memory,
        prompt_type=ptype,
        prompt_lang=plang,
    )


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    st.subheader("🔑 API Keys")
    groq_key   = st.text_input("Groq API Key",   type="password",
                                value=os.getenv("GROQ_API_KEY", ""),
                                help="Get free key at console.groq.com")
    gemini_key = st.text_input("Gemini API Key", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""),
                                help="Get free key at aistudio.google.com")

    st.markdown("---")
    st.subheader("🤖 LLM Selection")
    llm_name = st.selectbox(
        "Primary LLM",
        ["groq-llama", "groq-gemma", "gemini"],
        index=0,
        help="Groq (llama-3.1-8b) is primary; Groq/Gemini fallback auto-activates on failure.",
    )

    st.markdown("---")
    st.subheader("🧠 Memory Strategy")
    memory = st.selectbox(
        "Conversation Memory",
        ["sliding_window", "full_history", "strict_truncation", "summarized_history"],
        index=0,
        help=(
            "sliding_window: last 3 turns | "
            "full_history: all turns | "
            "strict_truncation: only last turn | "
            "summarized_history: LLM-compressed older turns"
        ),
    )
    window_size = st.slider("Window size (turns)", 1, 6, 3,
                             disabled=(memory != "sliding_window"))

    st.markdown("---")
    st.subheader("📝 Prompt Engineering")
    prompt_type = st.selectbox(
        "Prompt Type",
        ["system_guided", "minimal"],
        index=0,
        help="system_guided: strict grounding rules | minimal: bare-bones template",
    )
    prompt_lang = st.selectbox(
        "Prompt Language",
        ["en", "ar"],
        index=0,
        help="Language of the system instructions (not the user query)",
    )

    st.markdown("---")
    st.subheader("📚 Episodes in Index")
    st.markdown("""
- 🐙 الأخطبوط (Octopus)
- ⚔️ الساموراي (Samurai)
- 🕌 تاج محل (Taj Mahal)
- ⚛️ فيزياء الحركة (Physics)
- 🎬 Citizen Kane
""")

    st.markdown("---")
    force_rebuild = st.checkbox("Force rebuild index", value=False)
    load_btn = st.button("🔄 Load / Rebuild Index", use_container_width=True)

# ── main area tabs ───────────────────────────────────────────────────────────
tab_chat, tab_logs, tab_info = st.tabs(["💬 Chat", "📋 Logs", "ℹ️ System Info"])

# ── session state ─────────────────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "bot"        not in st.session_state: st.session_state.bot        = None
if "rag"        not in st.session_state: st.session_state.rag        = None
if "all_logs"   not in st.session_state: st.session_state.all_logs   = []
if "config_key" not in st.session_state: st.session_state.config_key = ""

current_config = f"{groq_key[:6]}{gemini_key[:6]}{llm_name}{memory}{prompt_type}{prompt_lang}"

# ── load index when button pressed or config changed ─────────────────────────
if load_btn or (st.session_state.rag is None and (groq_key or gemini_key)):
    if not groq_key and not gemini_key:
        st.sidebar.error("Please enter at least one API key.")
    else:
        with st.spinner("Building/loading index…"):
            rag = load_rag(groq_key, gemini_key, force_rebuild)
            st.session_state.rag = rag
            st.session_state.bot = make_bot(
                rag, groq_key, gemini_key, llm_name, memory, prompt_type, prompt_lang
            )
            st.session_state.config_key = current_config
        st.sidebar.success("Index ready ✓")

# Recreate bot when settings change (not the index)
elif (st.session_state.rag is not None and
      st.session_state.config_key != current_config):
    st.session_state.bot = make_bot(
        st.session_state.rag, groq_key, gemini_key,
        llm_name, memory, prompt_type, prompt_lang
    )
    st.session_state.config_key = current_config
    st.session_state.messages = []   # reset chat on config change


# ── CHAT TAB ─────────────────────────────────────────────────────────────────
with tab_chat:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("🎓 الدحيح RAG Chatbot")
        st.caption("Arabic multi-turn QA grounded in Al-Dahih episode transcripts.")
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.bot:
                st.session_state.bot.reset()

    # Display chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            if msg.get("is_ood"):
                st.markdown(f'<span class="ood-msg">{msg["content"]}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

            # show retrieved sources if available
            if msg.get("sources"):
                with st.expander("📌 Retrieved Sources", expanded=False):
                    for ep, score in msg["sources"]:
                        st.markdown(
                            f'<span class="source-badge">{ep}</span> '
                            f'<span class="score-badge">sim={score:.3f}</span>',
                            unsafe_allow_html=True,
                        )
                    if msg.get("context"):
                        st.text_area("Context sent to LLM", msg["context"],
                                     height=180, disabled=True)

    # Chat input
    if st.session_state.bot is None:
        st.info("👈 Enter your API key(s) in the sidebar and click **Load / Rebuild Index**.")
    else:
        user_input = st.chat_input("Ask about الأخطبوط، الساموراي، تاج محل، الفيزياء، or Citizen Kane…")
        if user_input:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving & generating…"):
                    t0     = time.time()
                    result = st.session_state.bot.chat(user_input)
                    elapsed = time.time() - t0

                if result["is_ood"]:
                    st.markdown(
                        f'<span class="ood-msg">{result["response"]}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(result["response"])

                # Source details
                if result.get("sources"):
                    with st.expander("📌 Retrieved Sources", expanded=True):
                        for ep, score in result["sources"]:
                            st.markdown(
                                f'<span class="source-badge">{ep}</span> '
                                f'<span class="score-badge">sim={score:.3f}</span>',
                                unsafe_allow_html=True,
                            )
                        if result.get("retrieved_context"):
                            st.text_area("Context sent to LLM",
                                         result["retrieved_context"],
                                         height=200, disabled=True)

                st.caption(f"⏱ {elapsed:.2f}s | LLM: {llm_name} | Memory: {memory} | Prompt: {prompt_type}_{prompt_lang}")

            # Save to session
            st.session_state.messages.append({
                "role":    "assistant",
                "content": result["response"],
                "sources": result.get("sources", []),
                "context": result.get("retrieved_context", ""),
                "is_ood":  result["is_ood"],
            })
            st.session_state.all_logs.extend(
                st.session_state.bot.logs[-1:]
            )


# ── LOGS TAB ─────────────────────────────────────────────────────────────────
with tab_logs:
    st.subheader("📋 System Logs")

    logs = st.session_state.all_logs
    if not logs:
        st.info("No logs yet – start chatting to generate logs.")
    else:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("💾 Save logs to evaluation_logs.json"):
                with open("evaluation_logs.json", "w", encoding="utf-8") as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
                st.success("Saved evaluation_logs.json")
        with col_b:
            st.metric("Total turns logged", len(logs))

        for i, log in enumerate(reversed(logs)):
            with st.expander(
                f"Turn {log.get('turn','?')} | "
                f"{'⚠️ OOD' if log.get('is_ood') else '✅'} | "
                f"score={log.get('max_score', 0):.3f} | "
                f"{log.get('llm','?')} / {log.get('memory','?')}",
                expanded=(i == 0),
            ):
                st.markdown(f'<div class="log-entry">'
                             f'<b>Query:</b> {log.get("query","")}<br>'
                             f'<b>Sources:</b> {", ".join(log.get("sources",[]))}<br>'
                             f'<b>Prompt:</b> {log.get("prompt_type","?")}_{log.get("prompt_lang","?")}<br>'
                             f'<b>Response (truncated):</b> {str(log.get("response",""))[:200]}…'
                             f'</div>', unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download full logs as JSON",
            data=json.dumps(logs, ensure_ascii=False, indent=2),
            file_name="evaluation_logs.json",
            mime="application/json",
        )


# ── INFO TAB ─────────────────────────────────────────────────────────────────
with tab_info:
    st.subheader("ℹ️ System Architecture")
    st.markdown("""
### RAG Pipeline Overview

| Component | Choice | Justification |
|---|---|---|
| **Embedding model** | `paraphrase-multilingual-MiniLM-L12-v2` | Covers 50+ langs incl. Arabic; preserves code-switching |
| **Vector store** | FAISS IndexFlatIP | Cosine sim via inner product; no server needed |
| **Chunking** | 250 words / 50 overlap | ~1–2 coherent paragraphs; prevents answer splits |
| **OOD threshold** | cosine sim < 0.25 | Empirically: in-domain ≥ 0.35, OOD < 0.20 |
| **Primary LLM** | Groq llama-3.1-8b-instant | Fast, free tier, Arabic + English support |
| **Fallback LLM** | Gemini 1.5-flash | Free tier, strong multilingual generation |
| **Retry policy** | 3 attempts, exponential backoff | Handles transient API failures gracefully |

### Episodes (5 selected from MS1)
- 🐙 **الأخطبوط** – Octopus biology and intelligence
- ⚔️ **الساموراي** – Japanese warrior culture
- 🕌 **تاج محل** – Taj Mahal history
- ⚛️ **فيزياء الحركة** – Physics and philosophy of motion
- 🎬 **Citizen Kane** – Film history and analysis

### Memory Strategies
| Strategy | Description | Best for |
|---|---|---|
| `sliding_window` | Last 3 Q/A pairs | Balanced cost + coherence |
| `full_history` | All turns | Long analytical conversations |
| `strict_truncation` | Only last Q/A pair | Minimal token cost |
| `summarized_history` | LLM-compressed older turns | Long sessions |

### Prompt Variants Evaluated
- `system_guided_en` – Detailed grounding rules, English instructions
- `minimal_en` – Bare-bones template, English
- `system_guided_ar` – Full rules, Arabic instructions
- `minimal_ar` – Bare-bones, Arabic
""")

    if st.session_state.rag:
        rag = st.session_state.rag
        st.markdown("---")
        st.subheader("📊 Index Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total chunks", len(rag.chunks))
        c2.metric("Episodes indexed", len(rag.EPISODES))
        if rag.faiss_index:
            c3.metric("FAISS vectors", rag.faiss_index.ntotal)

        # Chunk distribution per episode
        from collections import Counter
        import pandas as pd
        counts = Counter(c.source for c in rag.chunks)
        df = pd.DataFrame(counts.items(), columns=["Episode", "Chunks"])
        st.bar_chart(df.set_index("Episode"))
