# lg_app/src/app/graph.py
import os, re, json, logging, warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Vector DB
from qdrant_client import QdrantClient, models as qmodels
from urllib.parse import urlparse, urlunparse

# LangChain bits
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LangGraph
from langgraph.graph import StateGraph, START, END, MessagesState

# HF Inference for remote embeddings
from huggingface_hub import InferenceClient

try:
    # uses your remote/OpenAI-compatible llm + guard + BGE helpers
    from .llm_loader import generate_text, ask_secure, bge_sentence_similarity
except ImportError:
    from telco_cyber_chat.llm_loader import generate_text, ask_secure, bge_sentence_similarity

# ===================== Config / Secrets =====================
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_Cyber_Chat")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in env (use .env or Studio secrets).")

DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "it_specialist")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("LLM_TOKEN")
BGE_MODEL_ID = os.getenv("BGE_MODEL_ID", "BAAI/bge-m3")

# Kept for metadata only (orchestration runs via generate_text)
ORCH_MODEL_ID = (
    os.getenv("HF_MODEL_ID")
    or os.getenv("REMOTE_MODEL_ID")
    or "fdtn-ai/Foundation-Sec-8B-Instruct"
)

# Orchestration knobs
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "3"))
AGENT_TOPK_PER_STEP = int(os.getenv("AGENT_TOPK_PER_STEP", "4"))
AGENT_FORCE_FIRST_SEARCH = os.getenv("AGENT_FORCE_FIRST_SEARCH", "true").lower() == "true"
AGENT_MIN_STEPS = int(os.getenv("AGENT_MIN_STEPS", "1"))

RERANK_KEEP_TOPK = int(os.getenv("RERANK_KEEP_TOPK", "8"))
RERANK_PASS_THRESHOLD = float(os.getenv("RERANK_PASS_THRESHOLD", "0.25"))  # avg top scores to pass


# ===================== Qdrant helpers =====================
def _normalize_qdrant_url(raw: str) -> str:
    u = urlparse(raw)
    scheme = u.scheme or "https"
    netloc = u.netloc or u.path
    if scheme == "https" and ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return urlunparse((scheme, netloc, "", "", "", ""))

def _make_qdrant_client(url: str, api_key: Optional[str]) -> QdrantClient:
    url_norm = _normalize_qdrant_url(url)
    client = QdrantClient(url=url_norm, api_key=api_key, timeout=15.0)
    try:
        _ = client.get_collections()
    except Exception as e:
        u = urlparse(url_norm)
        if u.scheme == "http" and ":" not in u.netloc:
            client = QdrantClient(host=u.hostname, port=6333, api_key=api_key, https=False, timeout=15.0)
            _ = client.get_collections()
        else:
            raise RuntimeError(
                f"Could not reach Qdrant at '{url}'. Normalized: '{url_norm}'. "
                "For Qdrant Cloud use HTTPS without ':6333'. "
                f"Original error: {repr(e)}"
            )
    return client


# ===================== Remote BGE embeddings (no local weights) =====================
_bge_client: Optional[InferenceClient] = None

def _get_bge_client() -> InferenceClient:
    global _bge_client
    if _bge_client is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN is required for remote embeddings")
        # Standard signature: pass token; model is specified per call
        _bge_client = InferenceClient(token=HF_TOKEN)
    return _bge_client

def _embed_dense_many(texts: List[str]) -> List[List[float]]:
    cli = _get_bge_client()
    embs = cli.feature_extraction(texts, model=BGE_MODEL_ID)
    arr = np.array(embs)
    if arr.ndim == 1:
        arr = arr[None, :]
    # L2-normalize to match normalize_embeddings=True
    arr = arr / (np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-12)
    return arr.tolist()


# ===================== Clients =====================
qdrant = _make_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
_ = qdrant.scroll(collection_name=QDRANT_COLLECTION, limit=1, with_payload=False)


# ===================== Lightweight LLM helper (used by agents, not orchestrator) =====================
def _llm(prompt: str, max_new_tokens: int = 192) -> str:
    return generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1, top_p=1.0,
        return_full_text=False
    )


# ===================== Retrieval =====================
@dataclass
class RetrievalCfg:
    top_k: int = 6
    rrf_k: int = 60
    alpha_dense: float = 0.6
    overfetch: int = 3
    dense_name: str = "dense"
    text_key: str = "node_content"
    source_key: str = "node_id"

CFG = RetrievalCfg()

def _encode_dense(q: str):
    return _embed_dense_many([q])[0]

def _search_dense(q: str, k: int):
    vec = _encode_dense(q)
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qmodels.Query(vector=qmodels.NamedVector(name=CFG.dense_name, vector=vec)),
            limit=k, with_payload=True, with_vectors=False,
        ); return resp.points
    except Exception:
        pass
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            vector=qmodels.NamedVector(name=CFG.dense_name, vector=vec),
            limit=k, with_payload=True, with_vectors=False,
        ); return resp.points
    except Exception:
        pass
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            vector=vec, limit=k, with_payload=True, with_vectors=False,
        ); return resp.points
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return qdrant.search(collection_name=QDRANT_COLLECTION,
                                 query_vector=(CFG.dense_name, vec),
                                 limit=k, with_payload=True, with_vectors=False)
        except Exception:
            return qdrant.search(collection_name=QDRANT_COLLECTION,
                                 query_vector=vec,
                                 limit=k, with_payload=True, with_vectors=False)

def _rrf_fuse(dense_hits, other_hits, k_rrf: int, alpha_dense: float):
    # other_hits placeholder (sparse/web) not used here; keep RRF scaffold for future
    def rankmap(h): return {str(x.id): r for r, x in enumerate(h, 1)}
    rd, rs = rankmap(dense_hits), rankmap(other_hits or [])
    ids = set(rd) | set(rs)
    fused = []
    for pid in ids:
        sd = 1.0 / (k_rrf + rd.get(pid, 10**6))
        ss = 1.0 / (k_rrf + rs.get(pid, 10**6)) if rs else 0.0
        score = alpha_dense * sd + (1.0 - alpha_dense) * ss
        hit = next((h for h in dense_hits if str(h.id) == pid), None) or \
              next((h for h in (other_hits or []) if str(h.id) == pid), None)
        fused.append((score, hit))
    fused.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in fused]

def _to_docs(points) -> List[Document]:
    docs = []
    for i, h in enumerate(points, 1):
        pl = h.payload or {}
        point_id = str(getattr(h, "id", f"doc{i}"))
        src = pl.get(CFG.source_key, "")
        docs.append(
            Document(
                page_content=pl.get(CFG.text_key, "") or "",
                metadata={
                    "doc_id": point_id,
                    "source": src,
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                }
            )
        )
    return docs

def hybrid_search(q: str, top_k: int = None) -> List[Document]:
    k = top_k or CFG.top_k
    d = _search_dense(q, k * CFG.overfetch)
    fused = _rrf_fuse(d, None, CFG.rrf_k, CFG.alpha_dense)[:k]
    return _to_docs(fused)


# ===================== Graph state / helpers =====================
class ChatState(MessagesState):
    # MessagesState already defines:
    # messages: Annotated[list[AnyMessage], add_messages]
    query: str
    intent: str          # optional semantic label if needed later
    docs: List[Document]
    answer: str
    eval: Dict[str, Any]
    trace: List[str]

# 🔧 NEW: robust coercion to avoid list.strip errors
def _coerce_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_coerce_str(e) for e in x if e is not None)
    if x is None:
        return ""
    return str(x)

def _last_user(state: "ChatState") -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return _coerce_str(getattr(msg, "content", "")).strip()
    return _coerce_str(state.get("query", "")).strip()

def _fmt_ctx(docs: List[Document], cap: int = 12) -> str:
    out = []
    for i, d in enumerate(docs[:cap], 1):
        did = d.metadata.get("doc_id") or d.metadata.get("source") or f"doc{i}"
        chunk = _coerce_str(d.page_content).strip()
        out.append(f"[{did}] {chunk[:1200]}")
    return "\n\n".join(out) if out else "No context."

def _apply_rerank_for_query(docs: List[Document], q: str, keep: int = RERANK_KEEP_TOPK) -> List[Document]:
    if not docs:
        return []
    texts = [ _coerce_str(d.page_content)[:1024] for d in docs ]
    scores = bge_sentence_similarity(q, texts)  # remote similarity/rerank
    ranked = sorted(zip(docs, scores), key=lambda t: float(t[1]), reverse=True)
    for d, s in ranked:
        d.metadata["rerank_score"] = float(s)
    return [d for d, _ in ranked][:keep]

def _avg_rerank(docs: List[Document], k: int = 5) -> float:
    if not docs:
        return 0.0
    vals = []
    for d in docs[:k]:
        s = d.metadata.get("rerank_score", None)
        if s is not None:
            vals.append(float(s))
    return (sum(vals)/len(vals)) if vals else 0.0

def _infer_role(intent: str) -> str:
    if intent == "policy": return "admin"
    if intent in ("diagnostic", "incident", "mitigation"): return "network_admin"
    return DEFAULT_ROLE


# ===================== Orchestrator (classify & route) — via generate_text =====================
# Chat prompt that forces STRICT JSON
CLASSIFY_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are a strict classifier for a telecom-cyber RAG orchestrator.\n"
            "Return ONLY a single JSON object with keys: intent, clarity.\n"
            "intent ∈ {informational, diagnostic, policy, general}.\n"
            "clarity ∈ {clear, vague, multi-hop, longform}.\n"
            "No prose. No markdown. JSON only."
        ),
    ),
    (
        "human",
        "User: {q}\n\nRespond with JSON now.",
    ),
])

# Utility: flatten ChatMessages -> plain text prompt for generate_text
def _messages_to_text(msgs) -> str:
    parts = []
    # allow PromptValue passthrough
    if hasattr(msgs, "to_messages"):
        msgs = msgs.to_messages()
    for m in msgs:
        role = getattr(m, "type", "user").upper()
        parts.append(f"{role}:\n{_coerce_str(getattr(m, 'content', ''))}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)

_orch_chain = (
    CLASSIFY_CHAT_PROMPT
    | RunnableLambda(lambda pv: generate_text(
        _messages_to_text(pv),
        max_new_tokens=256,
        temperature=0.0,
        top_p=1.0
    ))
    | StrOutputParser()
)

# Utility: extract first JSON object from a string
_JSON_RE = re.compile(r"\{[\s\S]*?\}")

def orchestrator_node(state: ChatState) -> Dict:
    q = state.get("query") or _last_user(state)
    q = _coerce_str(q)

    try:
        raw = _orch_chain.invoke({"q": q})  # raw is a string from StrOutputParser
        m = _JSON_RE.search(raw)
        obj = json.loads(m.group(0) if m else raw)
        intent  = _coerce_str(obj.get("intent", "general")).lower()
        clarity = _coerce_str(obj.get("clarity", "clear")).lower()
    except Exception:
        intent, clarity = "general", "clear"

    role = _infer_role(intent)
    ev = dict(state.get("eval") or {})
    ev.update({
        "intent": intent,
        "clarity": clarity,
        "role": role,
        "orch_model": ORCH_MODEL_ID,
        "orch_steps": int(ev.get("orch_steps", 0)) + 1,
    })

    # Choose reasoning path
    if clarity == "clear":
        nxt = "react"
    elif clarity in ("multi-hop", "longform"):
        nxt = "self_ask"
    else:
        # vague/uncertain: default to self-ask for expansion
        nxt = "self_ask"

    return {
        "query": q,
        "intent": intent,
        "eval": ev,
        "trace": state.get("trace", []) + [f"orchestrator(intent={intent},clarity={clarity},role={role})->{nxt}"],
    }

def route_orchestrator(state: ChatState) -> str:
    clarity = (state.get("eval") or {}).get("clarity", "clear")
    return "react" if clarity == "clear" else "self_ask"


# ===================== Agents (each does internal hybrid retrieval) =====================
REACT_STEP_PROMPT = """
You are a ReAct telecom-cyber analyst. Output a suggestion for the next retrieval query.
If you include JSON, prefer: {{"action":"search","query":"<short query>","note":"<why>"}}.
But any format is allowed; I will parse heuristically.

User:
{question}

Snippets (trimmed):
{snips}
""".strip()

def _ctx_snips(docs: List[Document], cap: int = 3, width: int = 300) -> str:
    chunks = [ _coerce_str(d.page_content)[:width] for d in (docs or [])[:cap] ]
    return "\n---\n".join(chunks) if chunks else "(none)"

# ---- Heuristic parser: tolerate non-JSON and code fences ----
_ACTION_RE = re.compile(r'"?\baction\b"?\s*:\s*"?([A-Za-z_ \-]+)"?', re.I)
_QUERY_RE  = re.compile(r'"?\bquery\b"?\s*:\s*"([^"]+)"', re.I | re.S)
_CODEBLOCK = re.compile(r"```(?:json)?(.*?)```", re.S)

def _extract_action_query(raw: str) -> Tuple[Optional[str], Optional[str]]:
    if not raw:
        return None, None
    # prefer inside a code block if present
    mcb = _CODEBLOCK.search(raw)
    text = mcb.group(1) if mcb else raw

    am = _ACTION_RE.search(text)
    qm = _QUERY_RE.search(text)

    action = am.group(1).strip().lower() if am else None
    query  = qm.group(1).strip() if qm else None

    if action not in ("search", "finish"):
        action = None
    return action, query

def react_loop_node(state: ChatState) -> Dict:
    q = state["query"]
    ev = dict(state.get("eval") or {})
    step = int(ev.get("react_step", 0))
    docs = state.get("docs", []) or []
    snips = _ctx_snips(docs)

    raw = _llm(REACT_STEP_PROMPT.format(question=q, snips=snips)) or ""
    ev.setdefault("debug", {})["react_raw"] = raw[:800]  # keep a peek for debugging

    action, subq_extracted = _extract_action_query(raw)

    # Hard defaults to avoid KeyError no matter what the model returns
    action = (action or "search").strip().lower()
    if action not in ("search", "finish"):
        action = "search"
    subq   = (subq_extracted or q).strip() or q

    # Ensure at least one hop; and if first step & not 'search', force 'search'
    if (step < AGENT_MIN_STEPS) or (AGENT_FORCE_FIRST_SEARCH and step == 0 and action != "search"):
        action = "search"

    if action == "search":
        hop_docs = hybrid_search(subq, top_k=AGENT_TOPK_PER_STEP)
        docs = docs + hop_docs
        ev.setdefault("queries", []).append(subq)

    # bump step + last agent tag
    ev["react_step"] = step + 1
    ev["last_agent"] = "react"

    # From here, ALWAYS go to reranker
    return {
        "docs": docs,
        "eval": ev,
        "trace": state.get("trace", []) + [f"react_step({step}, action={action}, subq='{subq}') -> rerank[{len(docs)} docs]"]
    }

SELFASK_PLAN_PROMPT = """
Decompose the user question into 2-4 minimal, ordered sub-questions for multi-hop reasoning.
Return a JSON list only.
User: {question}
""".strip()

def self_ask_loop_node(state: ChatState) -> Dict:
    q = state["query"]
    ev = dict(state.get("eval") or {})
    subqs = ev.get("selfask_subqs")
    idx = int(ev.get("selfask_idx", 0))
    docs = state.get("docs", []) or []

    if not subqs:
        out = _llm(SELFASK_PLAN_PROMPT.format(question=q))
        try:
            arr = json.loads(re.search(r"\[[\s\S]*\]", out).group(0))
            subqs = [str(x) for x in arr if isinstance(x, (str, int, float))]
        except Exception:
            subqs = [q]
        ev["selfask_subqs"] = subqs
        idx = 0

    # take one hop this round
    subq = subqs[min(idx, max(0, len(subqs)-1))]
    hop_docs = hybrid_search(subq, top_k=AGENT_TOPK_PER_STEP)
    docs = docs + hop_docs

    ev.setdefault("queries", []).append(subq)
    ev["selfask_idx"] = idx + 1
    ev["last_agent"] = "self_ask"

    return {
        "docs": docs,
        "eval": ev,
        "trace": state.get("trace", []) + [f"self_ask_step({idx} '{subq}') -> rerank[{len(docs)} docs]"]
    }


# ===================== Reranker (pass/fail gating) =====================
def reranker_node(state: ChatState) -> Dict:
    q = state["query"]
    ev = dict(state.get("eval") or {})
    docs = state.get("docs", []) or []

    if not docs:
        # nothing to rank; force another hop if budget allows
        return {
            "eval": ev,
            "trace": state.get("trace", []) + ["rerank(EMPTY)"]
        }

    docs2 = _apply_rerank_for_query(docs, q, RERANK_KEEP_TOPK)
    avg_top = _avg_rerank(docs2, k=min(5, len(docs2)))
    ev["avg_rerank_top"] = float(avg_top)

    # budget checks per agent
    react_steps   = int(ev.get("react_step", 0))
    selfask_steps = int(ev.get("selfask_idx", 0))
    last_agent = ev.get("last_agent", "react")

    pass_gate = avg_top >= RERANK_PASS_THRESHOLD
    budget_ok = ((last_agent == "react" and react_steps < AGENT_MAX_STEPS) or
                 (last_agent == "self_ask" and selfask_steps < AGENT_MAX_STEPS))

    # Decide where to go next
    decision = "final" if pass_gate or not budget_ok else ("retry_react" if last_agent == "react" else "retry_self_ask")

    return {
        "docs": docs2,
        "eval": ev,
        "trace": state.get("trace", []) + [f"rerank(avg={avg_top:.3f}, keep={len(docs2)}) -> {decision}"],
    }

def route_rerank(state: ChatState) -> str:
    ev = state.get("eval") or {}
    last = ev.get("last_agent", "react")
    avg_top = float(ev.get("avg_rerank_top", 0.0))
    react_steps   = int(ev.get("react_step", 0))
    selfask_steps = int(ev.get("selfask_idx", 0))

    pass_gate = avg_top >= RERANK_PASS_THRESHOLD
    budget_ok = ((last == "react" and react_steps < AGENT_MAX_STEPS) or
                 (last == "self_ask" and selfask_steps < AGENT_MAX_STEPS))

    if pass_gate or not budget_ok:
        return "final"
    return "retry_react" if last == "react" else "retry_self_ask"


# ===================== LLM (final answer) =====================
def llm_node(state: ChatState) -> Dict:
    docs = state.get("docs", [])
    role = (state.get("eval") or {}).get("role") or DEFAULT_ROLE
    if not docs:
        msg = (
            f"No evidence found in Qdrant collection '{QDRANT_COLLECTION}'. I won't fabricate an answer.\n\n"
            "Troubleshooting:\n"
            "- Verify the collection has points (and correct payload keys)\n"
            "- Ensure the dense vector name matches your collection\n"
            "- Ensure embeddings match BGE-M3 (dim=1024)"
        )
        return {"messages":[AIMessage(content=msg)], "answer": msg,
                "trace": state.get("trace", []) + ["llm(NO_CONTEXT)"]}
    text = ask_secure(state["query"], context=_fmt_ctx(docs, cap=12), role=role,
                      preset="factual", max_new_tokens=400)
    return {"messages": [AIMessage(content=text)], "answer": text, "trace": state.get("trace", []) + ["llm"]}


# ===================== Graph wiring =====================
state_graph = StateGraph(ChatState)

# Core nodes
state_graph.add_node("orchestrator",  orchestrator_node)
state_graph.add_node("react_loop",    react_loop_node)
state_graph.add_node("self_ask_loop", self_ask_loop_node)
state_graph.add_node("rerank",        reranker_node)
state_graph.add_node("llm",           llm_node)

# Start → orchestrator
state_graph.add_edge(START, "orchestrator")

# Orchestrator chooses the agent
state_graph.add_conditional_edges("orchestrator", route_orchestrator, {
    "react": "react_loop",
    "self_ask": "self_ask_loop",
})

# Each agent does one hop then ALWAYS → rerank
state_graph.add_edge("react_loop", "rerank")
state_graph.add_edge("self_ask_loop", "rerank")

# Reranker either loops back to the SAME agent or forwards to LLM
state_graph.add_conditional_edges("rerank", route_rerank, {
    "retry_react": "react_loop",
    "retry_self_ask": "self_ask_loop",
    "final": "llm",
})

# LLM → END
state_graph.add_edge("llm", END)

# Export compiled graph for Studio
graph = state_graph.compile()
