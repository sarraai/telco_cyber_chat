import os, re, json, logging, warnings, hashlib
from dataclasses import dataclass
from functools import lru_cache
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

# --- HF Inference for remote BGE dense + our sparse lexicalizer (no local weights)
from huggingface_hub import InferenceClient

# Remote helpers (your LLM + HF sentence similarity)
try:
    from .llm_loader import generate_text, ask_secure, bge_sentence_similarity
except ImportError:
    from telco_cyber_chat.llm_loader import generate_text, ask_secure, bge_sentence_similarity

# ===================== Logging Configuration =====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format='%(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)

# ===================== Config / Secrets =====================
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in env (use .env or Studio secrets).")

DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "it_specialist")

# Vector names
DENSE_NAME  = os.getenv("DENSE_FIELD", "dense")
SPARSE_NAME = os.getenv("SPARSE_FIELD", "sparse")

# BGE-M3 (REMOTE dense; sparse via token2id or hashing fallback)
BGE_MODEL_ID        = os.getenv("BGE_MODEL_ID", "BAAI/bge-m3")
BGE_TOKEN2ID_PATH   = os.getenv("BGE_TOKEN2ID_PATH", "").strip()
SPARSE_MAX_TERMS    = int(os.getenv("SPARSE_MAX_TERMS", "256"))
IDX_HASH_SIZE       = int(os.getenv("IDX_HASH_SIZE", str(2**20)))

# HF Inference client
HF_TOKEN        = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
HF_INF_PROVIDER = os.getenv("HF_INF_PROVIDER", "").strip()

# Kept for metadata only (orchestration runs via generate_text)
ORCH_MODEL_ID = (
    os.getenv("HF_MODEL_ID")
    or os.getenv("REMOTE_MODEL_ID")
    or "fdtn-ai/Foundation-Sec-8B-Instruct"
)

# Orchestration knobs
AGENT_MAX_STEPS          = int(os.getenv("AGENT_MAX_STEPS", "3"))
AGENT_TOPK_PER_STEP      = int(os.getenv("AGENT_TOPK_PER_STEP", "4"))
AGENT_FORCE_FIRST_SEARCH = os.getenv("AGENT_FORCE_FIRST_SEARCH", "true").lower() == "true"
AGENT_MIN_STEPS          = int(os.getenv("AGENT_MIN_STEPS", "1"))

RERANK_KEEP_TOPK       = int(os.getenv("RERANK_KEEP_TOPK", "8"))
RERANK_PASS_THRESHOLD  = float(os.getenv("RERANK_PASS_THRESHOLD", "0.25"))

# ===================== Greeting/Goodbye Detection =====================
GREETING_REPLY = "Hello! I'm your telecom-cybersecurity assistant."
GOODBYE_REPLY  = "Goodbye!"
THANKS_REPLY   = "You're welcome!"

# CRITICAL: Precise regex that matches ENTIRE string, not just start
_GREET_RE = re.compile(r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day))\s*[!.?]*\s*$", re.I)
_BYE_RE   = re.compile(r"^\s*(bye|goodbye|see\s+you|see\s+ya|thanks?\s*,?\s*bye|farewell)\s*[!.?]*\s*$", re.I)
_THANKS_RE = re.compile(r"^\s*(thanks|thank\s+you|thx|ty)\s*[!.?]*\s*$", re.I)

def _is_greeting(text: str) -> bool:
    """CRITICAL: Must match entire string, not just start"""
    return bool(_GREET_RE.search((text or "").strip()))

def _is_goodbye(text: str) -> bool:
    """CRITICAL: Must match entire string, not just start"""
    return bool(_BYE_RE.search((text or "").strip()))

def _is_thanks(text: str) -> bool:
    """Check for thank you messages"""
    return bool(_THANKS_RE.search((text or "").strip()))

def _is_smalltalk(text: str) -> bool:
    """Detect any small talk that should skip RAG"""
    return _is_greeting(text) or _is_goodbye(text) or _is_thanks(text)

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

qdrant = _make_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
_ = qdrant.scroll(collection_name=QDRANT_COLLECTION, limit=1, with_payload=False)

# ===================== Remote BGE dense + lexical sparse =====================
@lru_cache(maxsize=1)
def _get_hf_client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF token missing: set HF_TOKEN/HUGGINGFACEHUB_API_TOKEN in deployment environment.")
    kw = dict(api_key=HF_TOKEN, timeout=60)
    if HF_INF_PROVIDER:
        kw["provider"] = HF_INF_PROVIDER
    return InferenceClient(**kw)

def _embed_bge_remote(text: str) -> List[float]:
    arr = np.array(_get_hf_client().feature_extraction(text, model=BGE_MODEL_ID))
    vec = arr if arr.ndim == 1 else arr[0]
    n = np.linalg.norm(vec) + 1e-12
    return (vec / n).astype("float32").tolist()

@lru_cache(maxsize=1)
def _get_token2id() -> Dict[str, int]:
    if BGE_TOKEN2ID_PATH and os.path.exists(BGE_TOKEN2ID_PATH):
        try:
            with open(BGE_TOKEN2ID_PATH, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, dict) and "vocab" in mapping and isinstance(mapping["vocab"], dict):
                mapping = mapping["vocab"]
            return {str(k): int(v) for k, v in mapping.items()}
        except Exception as e:
            log.warning(f"[BGE] Failed to load token2id at {BGE_TOKEN2ID_PATH}: {e}")
    log.warning(
        "[BGE] No token2id mapping provided. Falling back to hashing buckets "
        f"(size={IDX_HASH_SIZE}). For best sparse recall, set BGE_TOKEN2ID_PATH "
        "to the same token2id.json used in ingestion."
    )
    return {}

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)?", re.U)

def _tokenize_query_simple(q: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(q or "") if t.strip()]

def _hash_idx(term: str) -> int:
    h = hashlib.sha1(term.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % IDX_HASH_SIZE

def _lexicalize_query(q: str) -> qmodels.SparseVector:
    toks = _tokenize_query_simple(q)
    if not toks:
        return qmodels.SparseVector(indices=[], values=[])

    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    max_tf = max(counts.values())

    items = [(t, (counts[t] ** 0.5) / (max_tf ** 0.5 + 1e-9)) for t in counts]
    if SPARSE_MAX_TERMS > 0 and len(items) > SPARSE_MAX_TERMS:
        items = sorted(items, key=lambda kv: kv[1], reverse=True)[:SPARSE_MAX_TERMS]

    token2id = _get_token2id()
    indices, values = [], []
    if token2id:
        for tok, w in items:
            tid = token2id.get(tok)
            if tid is not None:
                indices.append(int(tid))
                values.append(float(w))
    else:
        for tok, w in items:
            indices.append(_hash_idx(tok))
            values.append(float(w))

    return qmodels.SparseVector(indices=indices, values=values)

def _encode_query_bge(q: str) -> Tuple[List[float], qmodels.SparseVector]:
    dense_vec = _embed_bge_remote(q)
    sparse_vec = _lexicalize_query(q)
    return dense_vec, sparse_vec

# ===================== Retrieval =====================
@dataclass
class RetrievalCfg:
    top_k: int = 6
    rrf_k: int = 60
    alpha_dense: float = 0.6
    overfetch: int = 3
    dense_name: str = DENSE_NAME
    sparse_name: str = SPARSE_NAME
    text_key: str = "node_content"
    source_key: str = "node_id"

CFG = RetrievalCfg()

def _search_dense(q: str, k: int):
    dense_vec, _ = _encode_query_bge(q)
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qmodels.Query(vector=qmodels.NamedVector(name=CFG.dense_name, vector=dense_vec)),
            limit=k, with_payload=True, with_vectors=False,
        )
        return resp.points
    except Exception as e1:
        log.debug(f"Named dense search failed: {e1}")
    
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qmodels.Query(vector=dense_vec),
            limit=k, with_payload=True, with_vectors=False,
        )
        return resp.points
    except Exception as e2:
        log.debug(f"Unnamed dense search failed: {e2}")
    
    try:
        return qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=(CFG.dense_name, dense_vec),
            limit=k, with_payload=True, with_vectors=False
        )
    except Exception as e3:
        log.debug(f"Tuple dense search failed: {e3}")
    
    try:
        return qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=dense_vec,
            limit=k, with_payload=True, with_vectors=False
        )
    except Exception as e4:
        log.warning(f"All dense search methods failed: {e4}")
        return []

def _search_sparse(q: str, k: int):
    _, sparse_vec = _encode_query_bge(q)

    if not getattr(sparse_vec, "indices", None):
        return []

    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qmodels.Query(sparse_vector=qmodels.NamedSparseVector(
                name=CFG.sparse_name, vector=sparse_vec)),
            limit=k, with_payload=True, with_vectors=False,
        )
        return resp.points
    except Exception as e1:
        log.debug(f"Named sparse search failed: {e1}")
    
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qmodels.Query(sparse_vector=sparse_vec),
            limit=k, with_payload=True, with_vectors=False,
        )
        return resp.points
    except Exception as e2:
        log.debug(f"Unnamed sparse search failed: {e2}")
    
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=qmodels.Prefetch(
                query=qmodels.Query(sparse_vector=sparse_vec),
                limit=k,
            ),
            query=qmodels.Query(fusion=qmodels.Fusion.RRF),
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        return resp.points
    except Exception as e3:
        log.warning(f"All sparse search methods failed: {e3}")
        return []

def _rrf_fuse(dense_hits, sparse_hits, k_rrf: int, alpha_dense: float):
    def rankmap(h): return {str(x.id): r for r, x in enumerate(h, 1)}
    rd, rs = rankmap(dense_hits or []), rankmap(sparse_hits or [])
    ids = set(rd) | set(rs)
    fused = []
    for pid in ids:
        sd = 1.0 / (k_rrf + rd.get(pid, 10**6))
        ss = 1.0 / (k_rrf + rs.get(pid, 10**6))
        score = alpha_dense * sd + (1.0 - alpha_dense) * ss
        hit = next((h for h in (dense_hits or []) if str(h.id) == pid), None) or \
              next((h for h in (sparse_hits or []) if str(h.id) == pid), None)
        fused.append((score, hit))
    fused.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in fused]

def _to_docs(points) -> List[Document]:
    docs = []
    for i, h in enumerate(points or [], 1):
        pl = h.payload or {}
        point_id = str(getattr(h, "id", f"doc{i}"))
        src = pl.get(CFG.source_key, "")
        docs.append(
            Document(
                page_content=str(pl.get(CFG.text_key, "") or ""),
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
    s = _search_sparse(q, k * CFG.overfetch)
    fused = _rrf_fuse(d, s, CFG.rrf_k, CFG.alpha_dense)[:k]
    return _to_docs(fused)

# ===================== Graph state / helpers =====================
class ChatState(MessagesState):
    query: str
    intent: str
    docs: List[Document]
    answer: str
    eval: Dict[str, Any]
    trace: List[str]

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
    texts = [_coerce_str(d.page_content)[:1024] for d in docs]
    scores = bge_sentence_similarity(q, texts)
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

# ===================== Orchestrator (classify & route) =====================
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
    ("human", "User: {q}\n\nRespond with JSON now."),
])

def _messages_to_text(msgs) -> str:
    parts = []
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

_JSON_RE = re.compile(r"\{[\s\S]*?\}")

def orchestrator_node(state: ChatState) -> Dict:
    """
    CRITICAL FIXES:
    1. Check greeting/goodbye FIRST before any LLM classification
    2. Check for extremely short queries (1-2 words) and treat as small talk
    3. Prevent ReAct from running on small talk
    """
    q = state.get("query") or _last_user(state)
    q = _coerce_str(q).strip()
    
    log.info(f"[ORCHESTRATOR] Received query: '{q}'")
    
    # ========== LAYER 1: Check small talk BEFORE LLM classification ==========
    if _is_greeting(q):
        log.info(f"[ORCHESTRATOR] ✅ GREETING detected - SKIP ALL processing")
        return {
            "query": q,
            "intent": "greeting",
            "eval": {
                "intent": "greeting", 
                "clarity": "clear", 
                "role": DEFAULT_ROLE, 
                "skip_rag": True,
                "skip_reason": "smalltalk_greeting"
            },
            "messages": [AIMessage(content=GREETING_REPLY)],
            "answer": GREETING_REPLY,
            "docs": [],
            "trace": ["orchestrator(greeting)->END"]
        }
    
    if _is_goodbye(q):
        log.info(f"[ORCHESTRATOR] ✅ GOODBYE detected - SKIP ALL processing")
        return {
            "query": q,
            "intent": "goodbye",
            "eval": {
                "intent": "goodbye", 
                "clarity": "clear", 
                "role": DEFAULT_ROLE, 
                "skip_rag": True,
                "skip_reason": "smalltalk_goodbye"
            },
            "messages": [AIMessage(content=GOODBYE_REPLY)],
            "answer": GOODBYE_REPLY,
            "docs": [],
            "trace": ["orchestrator(goodbye)->END"]
        }
    
    if _is_thanks(q):
        log.info(f"[ORCHESTRATOR] ✅ THANKS detected - SKIP ALL processing")
        return {
            "query": q,
            "intent": "thanks",
            "eval": {
                "intent": "thanks", 
                "clarity": "clear", 
                "role": DEFAULT_ROLE, 
                "skip_rag": True,
                "skip_reason": "smalltalk_thanks"
            },
            "messages": [AIMessage(content=THANKS_REPLY)],
            "answer": THANKS_REPLY,
            "docs": [],
            "trace": ["orchestrator(thanks)->END"]
        }

    # ========== LAYER 2: Check for extremely short queries ==========
    # If query is 1-2 words and not a question, treat as small talk
    word_count = len(re.findall(r'\w+', q))
    if word_count <= 2 and not q.endswith('?'):
        log.info(f"[ORCHESTRATOR] ⚠️ Very short query ({word_count} words) - treating as small talk")
        return {
            "query": q,
            "intent": "smalltalk",
            "eval": {
                "intent": "smalltalk", 
                "clarity": "clear", 
                "role": DEFAULT_ROLE, 
                "skip_rag": True,
                "skip_reason": "too_short"
            },
            "messages": [AIMessage(content="I'm here to help with telecom and cybersecurity questions. What would you like to know?")],
            "answer": "I'm here to help with telecom and cybersecurity questions. What would you like to know?",
            "docs": [],
            "trace": ["orchestrator(smalltalk_short)->END"]
        }

    # ========== LAYER 3: Normal classification for technical queries ==========
    log.info(f"[ORCHESTRATOR] Technical query - proceeding with classification")
    try:
        raw = _orch_chain.invoke({"q": q})
        m = _JSON_RE.search(raw)
        obj = json.loads(m.group(0) if m else raw)
        intent  = _coerce_str(obj.get("intent", "general")).lower()
        clarity = _coerce_str(obj.get("clarity", "clear")).lower()
    except Exception as e:
        log.warning(f"[ORCHESTRATOR] Classification failed: {e}")
        intent, clarity = "general", "clear"

    role = _infer_role(intent)
    ev = dict(state.get("eval") or {})
    ev.update({
        "intent": intent,
        "clarity": clarity,
        "role": role,
        "orch_model": ORCH_MODEL_ID,
        "orch_steps": int(ev.get("orch_steps", 0)) + 1,
        "skip_rag": False,  # Explicitly mark as NOT skipping
    })

    nxt = "react" if clarity == "clear" else "self_ask"
    log.info(f"[ORCHESTRATOR] Routing to {nxt}")
    return {
        "query": q,
        "intent": intent,
        "eval": ev,
        "trace": state.get("trace", []) + [f"orchestrator(intent={intent},clarity={clarity},role={role})->{nxt}"],
    }

def route_orchestrator(state: ChatState) -> str:
    """
    CRITICAL FIX: Check skip_rag flag to bypass entire pipeline for greetings/goodbyes.
    This is the SECOND line of defense after orchestrator_node.
    """
    ev = state.get("eval") or {}
    
    # If skip_rag is set (greeting/goodbye/thanks), go directly to END
    if ev.get("skip_rag"):
        log.info("[ROUTE] skip_rag=True -> going to END immediately")
        return "end"
    
    # Otherwise route based on clarity
    clarity = ev.get("clarity", "clear")
    route = "react" if clarity == "clear" else "self_ask"
    log.info(f"[ROUTE] clarity={clarity} -> {route}")
    return route

# ===================== Agents =====================
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
    chunks = [_coerce_str(d.page_content)[:width] for d in (docs or [])[:cap]]
    return "\n---\n".join(chunks) if chunks else "(none)"

_ACTION_RE = re.compile(r'"?\baction\b"?\s*:\s*"?([A-Za-z_ \-]+)"?', re.I)
_QUERY_RE  = re.compile(r'"?\bquery\b"?\s*:\s*"([^"]+)"', re.I | re.S)
_CODEBLOCK = re.compile(r"```(?:json)?(.*?)```", re.S)

def _extract_action_query(raw: str) -> Tuple[Optional[str], Optional[str]]:
    if not raw:
        return None, None
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
    """
    CRITICAL FIX: Add safety checks to prevent hallucinated queries.
    1. Abort if small talk somehow reaches here
    2. Validate extracted subquery has word overlap with original
    3. Use original query if subquery seems hallucinated
    """
    q = state["query"]
    ev = dict(state.get("eval") or {})
    step = int(ev.get("react_step", 0))
    docs = state.get("docs", []) or []
    snips = _ctx_snips(docs)

    # ========== SAFETY CHECK 1: Don't process small talk ==========
    if _is_smalltalk(q):
        log.warning(f"[REACT] Small talk detected in react_loop! This should never happen. Query: '{q}'")
        return {
            "docs": [],
            "eval": ev,
            "trace": state.get("trace", []) + [f"react_step({step}) ABORTED - smalltalk detected"]
        }

    raw = generate_text(
        REACT_STEP_PROMPT.format(question=q, snips=snips),
        max_new_tokens=192, do_sample=False, num_beams=1, top_p=1.0, return_full_text=False
    ) or ""
    ev.setdefault("debug", {})["react_raw"] = raw[:800]

    action, subq_extracted = _extract_action_query(raw)
    action = (action or "search").strip().lower()
    if action not in ("search", "finish"):
        action = "search"
    
    # ========== SAFETY CHECK 2: Validate subquery makes sense ==========
    subq = (subq_extracted or q).strip() or q
    
    # If subquery was extracted, check it has word overlap with original
    if subq_extracted and subq_extracted != q:
        original_words = set(re.findall(r'\w+', q.lower()))
        subq_words = set(re.findall(r'\w+', subq.lower()))
        overlap = len(original_words & subq_words)
        
        # If no word overlap, likely hallucinated - use original query
        if overlap == 0 and len(original_words) > 0:
            log.warning(f"[REACT] Subquery has no overlap with original. Using original. Original: '{q}', Subquery: '{subq}'")
            subq = q

    if (step < AGENT_MIN_STEPS) or (AGENT_FORCE_FIRST_SEARCH and step == 0 and action != "search"):
        action = "search"

    if action == "search":
        hop_docs = hybrid_search(subq, top_k=AGENT_TOPK_PER_STEP)
        docs = docs + hop_docs
        ev.setdefault("queries", []).append(subq)

    ev["react_step"] = step + 1
    ev["last_agent"] = "react"

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
        out = generate_text(SELFASK_PLAN_PROMPT.format(question=q), max_new_tokens=160)
        try:
            arr = json.loads(re.search(r"\[[\s\S]*\]", out).group(0))
            subqs = [str(x) for x in arr if isinstance(x, (str, int, float))]
        except Exception:
            subqs = [q]
        ev["selfask_subqs"] = subqs
        idx = 0

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
        return { "eval": ev, "trace": state.get("trace", []) + ["rerank(EMPTY)"] }

    docs2 = _apply_rerank_for_query(docs, q, RERANK_KEEP_TOPK)
    avg_top = _avg_rerank(docs2, k=min(5, len(docs2)))
    ev["avg_rerank_top"] = float(avg_top)

    react_steps   = int(ev.get("react_step", 0))
    selfask_steps = int(ev.get("selfask_idx", 0))
    last_agent = ev.get("last_agent", "react")

    pass_gate = avg_top >= RERANK_PASS_THRESHOLD
    budget_ok = ((last_agent == "react" and react_steps < AGENT_MAX_STEPS) or
                 (last_agent == "self_ask" and selfask_steps < AGENT_MAX_STEPS))

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
            f"- Ensure dense name='{DENSE_NAME}' and sparse name='{SPARSE_NAME}' match your collection\n"
            "- Ensure embeddings match BGE-M3 (dense dim=1024) and sparse vocab mapping or hashing size"
        )
        return {"messages":[AIMessage(content=msg)], "answer": msg,
                "trace": state.get("trace", []) + ["llm(NO_CONTEXT)"]}

    text = ask_secure(state["query"], context=_fmt_ctx(docs, cap=12), role=role,
                      preset="factual", max_new_tokens=400)
    return {"messages": [AIMessage(content=text)], "answer": text, "trace": state.get("trace", []) + ["llm"]}

# ===================== Graph wiring =====================
state_graph = StateGraph(ChatState)

state_graph.add_node("orchestrator",  orchestrator_node)
state_graph.add_node("react_loop",    react_loop_node)
state_graph.add_node("self_ask_loop", self_ask_loop_node)
state_graph.add_node("rerank",        reranker_node)
state_graph.add_node("llm",           llm_node)

state_graph.add_edge(START, "orchestrator")

# CRITICAL FIX: Add "end" route that goes directly to END for small talk
state_graph.add_conditional_edges("orchestrator", route_orchestrator, {
    "react": "react_loop",
    "self_ask": "self_ask_loop",
    "end": END,  # Direct exit for greetings/goodbyes/thanks
})

state_graph.add_edge("react_loop", "rerank")
state_graph.add_edge("self_ask_loop", "rerank")
state_graph.add_conditional_edges("rerank", route_rerank, {
    "retry_react": "react_loop",
    "retry_self_ask": "self_ask_loop",
    "final": "llm",
})
state_graph.add_edge("llm", END)

graph = state_graph.compile()

# ===================== CRITICAL: Export with greeting pre-check =====================
def chat_with_greeting_precheck(query: str, **kwargs):
    """
    CRITICAL WRAPPER: Check for small talk BEFORE invoking graph.
    This is the FIRST line of defense and ensures instant responses.
    
    Use this function as your main entry point instead of graph.invoke()
    """
    q = query.strip()
    
    log.info(f"[PRE-CHECK] Query: '{q}'")
    
    # Fast-path Layer 1: Greetings
    if _is_greeting(q):
        log.info("[PRE-CHECK] ✅ GREETING - fast path")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=GREETING_REPLY)],
            "answer": GREETING_REPLY,
            "query": q,
            "intent": "greeting",
            "docs": [],
            "eval": {"intent": "greeting", "skip_reason": "pre_check"},
            "trace": ["pre_check_greeting"]
        }
    
    # Fast-path Layer 2: Goodbyes
    if _is_goodbye(q):
        log.info("[PRE-CHECK] ✅ GOODBYE - fast path")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=GOODBYE_REPLY)],
            "answer": GOODBYE_REPLY,
            "query": q,
            "intent": "goodbye",
            "docs": [],
            "eval": {"intent": "goodbye", "skip_reason": "pre_check"},
            "trace": ["pre_check_goodbye"]
        }
    
    # Fast-path Layer 3: Thanks
    if _is_thanks(q):
        log.info("[PRE-CHECK] ✅ THANKS - fast path")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=THANKS_REPLY)],
            "answer": THANKS_REPLY,
            "query": q,
            "intent": "thanks",
            "docs": [],
            "eval": {"intent": "thanks", "skip_reason": "pre_check"},
            "trace": ["pre_check_thanks"]
        }
    
    # Normal path: invoke graph for technical queries
    log.info("[PRE-CHECK] Technical query - invoking graph")
    initial_state = {
        "query": q,
        "messages": [HumanMessage(content=q)],
        "trace": []
    }
    return graph.invoke(initial_state)

# For backward compatibility, but recommend using chat_with_greeting_precheck
__all__ = ["graph", "chat_with_greeting_precheck", "hybrid_search"]
