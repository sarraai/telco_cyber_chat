# lg_app/src/app/graph.py
import os, re, json, logging, warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Annotated, Tuple

import torch
from typing_extensions import TypedDict

# Embeddings / reranker
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# Vector DB
from qdrant_client import QdrantClient, models as qmodels
from urllib.parse import urlparse, urlunparse

# LangChain bits
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph
from langgraph.graph import StateGraph, START, END, add_messages

try:
    from .llm_loader import generate_text, ask_secure
except ImportError:
    # fallback if someone runs files directly without installing the package
from telco_cyber_chat.llm_loader import generate_text, ask_secure


# ===================== Config / Secrets =====================
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_Cyber_Chat")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in env (use .env or Studio secrets).")

DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "it_specialist")

# Knobs
REWRITES_IF_UNCLEAR = 3
KEEP_TOPK_AFTER_RERANK = 8

AGENT_MAX_STEPS = 3
AGENT_TOPK_PER_STEP = 4
AGENT_FORCE_FIRST_SEARCH = True
AGENT_MIN_STEPS = 1
RUN_SHARED_RETRIEVER_EACH_STEP = True

ORCH_MAX_STEPS   = 6
ORCH_MIN_DOCS    = 4
ORCH_MIN_RERANK  = 0.25

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

# ===================== Models / Clients =====================
bge = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True if torch.cuda.is_available() else False,
    normalize_embeddings=True
)
bge_reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=True if torch.cuda.is_available() else False
)

qdrant = _make_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
_ = qdrant.scroll(collection_name=QDRANT_COLLECTION, limit=1, with_payload=False)

# ===================== Lightweight LLM helper =====================
def _llm(prompt: str, max_new_tokens: int = 192) -> str:
    return generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1, top_p=1.0,
        return_full_text=False
    )

# ===================== Hybrid retrieval =====================
@dataclass
class HybridCfg:
    top_k: int = 6
    rrf_k: int = 60
    alpha_dense: float = 0.6
    overfetch: int = 3
    dense_name: str = "dense"
    sparse_name: str = "sparse"
    text_key: str = "node_content"
    source_key: str = "node_id"

CFG = HybridCfg()

def _encode_dense(q: str):
    out = bge.encode([q], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    return out["dense_vecs"][0]

def _encode_sparse(q: str) -> qmodels.SparseVector:
    out = bge.encode([q], return_dense=False, return_sparse=True, return_colbert_vecs=False)
    lw = out["lexical_weights"][0]
    idxs = [int(i) for i in lw.keys()]
    vals = [float(lw[i]) for i in lw.keys()]
    return qmodels.SparseVector(indices=idxs, values=vals)

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

def _search_sparse(q: str, k: int):
    try:
        resp = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qmodels.Query(
                sparse=qmodels.NamedSparseVector(name=CFG.sparse_name, vector=_encode_sparse(q))
            ),
            limit=k, with_payload=True, with_vectors=False,
        )
        return resp.points
    except Exception:
        return []

def _rrf_fuse(dense_hits, sparse_hits, k_rrf: int, alpha_dense: float):
    def rankmap(h): return {str(x.id): r for r, x in enumerate(h, 1)}
    rd, rs = rankmap(dense_hits), rankmap(sparse_hits)
    ids = set(rd) | set(rs)
    fused = []
    for pid in ids:
        sd = 1.0 / (k_rrf + rd.get(pid, 10**6))
        ss = 1.0 / (k_rrf + rs.get(pid, 10**6)) if rs else 0.0
        score = alpha_dense * sd + (1.0 - alpha_dense) * ss
        hit = next((h for h in dense_hits if str(h.id) == pid), None) or \
              next((h for h in sparse_hits if str(h.id) == pid), None)
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
    s = _search_sparse(q, k * CFG.overfetch)
    if not d and not s:
        return []
    fused = _rrf_fuse(d, s, CFG.rrf_k, CFG.alpha_dense)[:k]
    return _to_docs(fused)

# ===================== Graph state / helpers =====================
class ChatState(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    intent: str
    docs: List[Document]
    answer: str
    eval: Dict[str, Any]
    trace: List[str]
    decision: str

def _fmt_ctx(docs: List[Document], cap: int = 12) -> str:
    out = []
    for i, d in enumerate(docs[:cap], 1):
        did = d.metadata.get("doc_id") or d.metadata.get("source") or f"doc{i}"
        chunk = (d.page_content or "").strip()
        out.append(f"[{did}] {chunk[:1200]}")
    return "\n\n".join(out) if out else "No context."

def _apply_rerank_for_query(docs: List[Document], q: str, keep: int = KEEP_TOPK_AFTER_RERANK) -> List[Document]:
    if not docs:
        return []
    pairs = [(q, (d.page_content or "")[:1024]) for d in docs]
    scores = bge_reranker.compute_score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda t: t[1], reverse=True)
    for d, s in ranked:
        d.metadata["rerank_score"] = float(s)
    return [d for d, _ in ranked][:keep]

def _answer_from_docs(question: str, docs: List[Document], role: str) -> str:
    context = _fmt_ctx(docs, cap=12)
    return ask_secure(
        question,
        context=context,
        role=role,
        preset="factual",
        max_new_tokens=400
    )

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
def _parse_json_obj(txt: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        m = _JSON_OBJ_RE.search(txt)
        if not m:
            return fallback
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return fallback

def _safe_json_list(out) -> Optional[List[str]]:
    txt = out if isinstance(out, str) else getattr(out, "content", str(out))
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [str(x) for x in obj if isinstance(x, (str, int, float))]
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", txt)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x) for x in obj if isinstance(x, (str, int, float))]
        except Exception:
            return None
    return None

def _last_user(state: ChatState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return (msg.content or "").strip()
    return (state.get("query") or "").strip()

def _infer_role(intent: str) -> str:
    if intent == "policy": return "admin"
    if intent in ("diagnostic", "incident", "mitigation"): return "network_admin"
    return DEFAULT_ROLE

# ===================== Nodes =====================
def router_node(state: ChatState) -> Dict:
    q = state["query"] if state.get("query") else _last_user(state)
    prompt = (
        "Classify the user query.\n"
        'Return JSON like {"intent":"informational|diagnostic|policy|general","clarity":"clear|vague|multi-hop|longform"}.\n'
        f"User: {q}"
    )
    try:
        txt = _llm(prompt)
        obj = json.loads(re.search(r"\{[\s\S]*\}", txt).group(0))
        intent = str(obj.get("intent","general")).lower()
        clarity = str(obj.get("clarity","clear")).lower()
    except Exception:
        intent, clarity = "general", "clear"
    role = _infer_role(intent)
    ev = dict(state.get("eval") or {})
    ev.update({"intent": intent, "clarity": clarity, "role": role})
    return {"query": q, "intent": intent, "eval": ev,
            "trace": state.get("trace", []) + [f"router({intent},{clarity},role={role})"]}

def query_rewriter_node(state: ChatState) -> Dict:
    q = state["query"]
    if REWRITES_IF_UNCLEAR <= 0:
        return {"trace": state.get("trace", []) + ["rewrite(SKIP)"]}
    out = _llm(f"Rewrite into {REWRITES_IF_UNCLEAR} concise search queries for retrieval. "
               "Return a JSON list only.\nUser: " + q)
    rewrites = _safe_json_list(out) or [q]
    ev = dict(state.get("eval") or {})
    ev["queries"] = rewrites
    return {"eval": ev, "trace": state.get("trace", []) + [f"rewrite({len(rewrites)})"]}

REACT_STEP_PROMPT = """
You are a ReAct telecom-cyber analyst. OUTPUT only JSON.
At step 0 you MUST return action="search" with a concise retrieval-ready query (unless trivial).
Strict JSON:
{"action":"search"|"finish","query":"<if search>","note":"<why>"}

User:
{question}

Snippets (trimmed):
{snips}
""".strip()

def _ctx_snips(docs: List[Document], cap: int = 3, width: int = 300) -> str:
    chunks = [ (d.page_content or "")[:width] for d in (docs or [])[:cap] ]
    return "\n---\n".join(chunks) if chunks else "(none)"

def react_loop_node(state: ChatState) -> Dict:
    q = state["query"]
    ev = dict(state.get("eval") or {})
    step = int(ev.get("react_step", 0))
    snips = _ctx_snips(state.get("docs", []))
    docs = state.get("docs", []) or []

    raw = _llm(REACT_STEP_PROMPT.format(question=q, snips=snips))
    obj = _parse_json_obj(raw, fallback={"action": "finish", "query": "", "note": "fallback"})
    action = str(obj.get("action", "finish")).lower()
    subq = (obj.get("query") or "").strip()

    if (step < AGENT_MIN_STEPS) or (AGENT_FORCE_FIRST_SEARCH and step == 0 and action != "search"):
        action = "search"
        if not subq:
            subq = q

    if step < AGENT_MAX_STEPS and action == "search":
        query_to_use = subq if subq else q
        hop_docs = hybrid_search(query_to_use, top_k=AGENT_TOPK_PER_STEP)
        docs = docs + hop_docs
        ev.setdefault("queries", []).append(query_to_use)

        if RUN_SHARED_RETRIEVER_EACH_STEP and ev.get("queries"):
            pooled = _multi_query_retrieve(ev["queries"], CFG.top_k)
            docs = docs + pooled

        ev["react_step"] = step + 1
        decision = "loop" if ev["react_step"] < AGENT_MAX_STEPS else "done"

        if decision == "done":
            docs2 = _apply_rerank_for_query(docs, q, KEEP_TOPK_AFTER_RERANK)
            role = ev.get("role") or DEFAULT_ROLE
            text = _answer_from_docs(q, docs2, role)
            return {"docs": docs2, "eval": ev, "decision": "done",
                    "messages": [AIMessage(content=text)], "answer": text,
                    "trace": state.get("trace", []) + [f"react_loop(step={step}, search->finish)"]}

        return {"docs": docs, "eval": ev, "decision": "loop",
                "trace": state.get("trace", []) + [f"react_loop(step={step}, search)"]}

    docs2 = _apply_rerank_for_query(docs, q, KEEP_TOPK_AFTER_RERANK)
    role = ev.get("role") or DEFAULT_ROLE
    text = _answer_from_docs(q, docs2, role)
    return {"docs": docs2, "eval": ev, "decision": "done",
            "messages": [AIMessage(content=text)], "answer": text,
            "trace": state.get("trace", []) + [f"react_loop(step={step}, finish)"]}

def route_react_loop(state: ChatState) -> str:
    return "loop" if (state.get("eval") or {}).get("react_step", 0) < AGENT_MAX_STEPS else "done" \
        if state.get("decision") == "done" else "loop"

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
        subqs = _safe_json_list(out) or [q]
        ev["selfask_subqs"] = subqs
        ev["selfask_idx"] = 0
        idx = 0

    max_hops = min(len(subqs), AGENT_MAX_STEPS)

    if idx < max_hops:
        subq = subqs[idx]
        hop_docs = hybrid_search(subq, top_k=AGENT_TOPK_PER_STEP)
        docs = docs + hop_docs

        ev.setdefault("queries", []).append(subq)
        ev["selfask_idx"] = idx + 1

        if RUN_SHARED_RETRIEVER_EACH_STEP and ev.get("queries"):
            pooled = _multi_query_retrieve(ev["queries"], CFG.top_k)
            docs = docs + pooled

        decision = "loop" if ev["selfask_idx"] < max_hops else "done"
        if decision == "done":
            docs2 = _apply_rerank_for_query(docs, q, KEEP_TOPK_AFTER_RERANK)
            role = ev.get("role") or DEFAULT_ROLE
            text = _answer_from_docs(q, docs2, role)
            return {"docs": docs2, "eval": ev, "decision": "done",
                    "messages": [AIMessage(content=text)], "answer": text,
                    "trace": state.get("trace", []) + [f"selfask_finish({idx})"]}

        return {"docs": docs, "eval": ev, "decision": "loop",
                "trace": state.get("trace", []) + [f"selfask_step({idx}->{ev['selfask_idx']})/{decision}"]}

    docs2 = _apply_rerank_for_query(docs, q, KEEP_TOPK_AFTER_RERANK)
    role = ev.get("role") or DEFAULT_ROLE
    text = _answer_from_docs(q, docs2, role)
    return {"docs": docs2, "eval": ev, "decision": "done",
            "messages": [AIMessage(content=text)], "answer": text,
            "trace": state.get("trace", []) + [f"selfask_finish({idx})"]}

def route_selfask_loop(state: ChatState) -> str:
    return "loop" if (state.get("eval") or {}).get("selfask_idx", 0) < AGENT_MAX_STEPS else "done" \
        if state.get("decision") == "done" else "loop"

def _multi_query_retrieve(queries: List[str], k: int) -> List[Document]:
    ranked_lists: List[List[Document]] = [hybrid_search(q, top_k=k) for q in queries]
    pool: Dict[str, Tuple[Document, float]] = {}
    for li, lst in enumerate(ranked_lists):
        for r, d in enumerate(lst, 1):
            did = d.metadata.get("doc_id") or f"doc-{li}-{r}"
            score = 1.0 / (CFG.rrf_k + r)
            if did not in pool or score > pool[did][1]:
                pool[did] = (d, score)
    fused = sorted(pool.values(), key=lambda t: t[1], reverse=True)
    return [d for d, _ in fused][:CFG.top_k]

def retriever_node(state: ChatState) -> Dict:
    q = state["query"]
    ev = dict(state.get("eval") or {})
    queries = ev.get("queries") or [q]
    existing = state.get("docs", []) or []
    new_docs = _multi_query_retrieve(queries, CFG.top_k)
    merged = existing + new_docs
    return {"docs": merged, "eval": ev, "trace": state.get("trace", []) + [f"retrieve({len(merged)})"]}

def reranker_node(state: ChatState) -> Dict:
    docs = state.get("docs", []) or []
    if not docs:
        return {"trace": state.get("trace", []) + ["rerank(EMPTY)"]}
    q = state["query"]
    docs2 = _apply_rerank_for_query(docs, q, KEEP_TOPK_AFTER_RERANK)
    return {"docs": docs2, "trace": state.get("trace", []) + [f"rerank({len(docs2)})"]}

def llm_node(state: ChatState) -> Dict:
    docs = state.get("docs", [])
    role = (state.get("eval") or {}).get("role") or DEFAULT_ROLE
    if not docs:
        msg = (
            f"No evidence found in Qdrant collection '{QDRANT_COLLECTION}'. I won't fabricate an answer.\n\n"
            "Troubleshooting:\n"
            "- Verify the collection has points (and correct payload keys)\n"
            "- Check CFG.dense_name matches your vector name, or that the collection is single-vector (unnamed)\n"
            "- Ensure your sparse index is configured as named 'sparse' in Qdrant\n"
            "- Ensure embeddings match BGE-M3 (dim=1024)"
        )
        return {"messages":[AIMessage(content=msg)], "answer": msg,
                "trace": state.get("trace", []) + ["llm(NO_CONTEXT)"]}
    text = _answer_from_docs(state["query"], docs, role)
    return {"messages": [AIMessage(content=text)], "answer": text, "trace": state.get("trace", []) + ["llm"]}

def _avg_rerank(docs: List[Document], k: int = 5) -> float:
    if not docs:
        return 0.0
    vals = []
    for d in docs[:k]:
        s = d.metadata.get("rerank_score", None)
        if s is not None: vals.append(float(s))
    return (sum(vals)/len(vals)) if vals else 0.0

def orchestrator_node(state: ChatState) -> Dict:
    ev   = dict(state.get("eval") or {})
    docs = state.get("docs", []) or []
    steps      = int(ev.get("orch_steps", 0))
    clarity    = ev.get("clarity", "clear")
    have_docs  = len(docs) >= ORCH_MIN_DOCS
    avg_score  = _avg_rerank(docs, k=min(5, len(docs)))
    budget_hit = steps >= ORCH_MAX_STEPS

    if steps == 0 and not have_docs:
        nxt = "retrieve"
    elif (not have_docs or avg_score < ORCH_MIN_RERANK) and steps < (ORCH_MAX_STEPS - 1):
        nxt = "retrieve"
    else:
        if clarity == "clear":
            nxt = "react"
        elif clarity in ("multi-hop", "longform"):
            nxt = "self_ask"
        else:
            nxt = "final"

    if budget_hit:
        nxt = "final"

    ev.update({
        "orch_steps": steps + 1,
        "next": nxt,
        "avg_rerank_top": avg_score,
        "have_docs": have_docs
    })
    return {"eval": ev, "trace": state.get("trace", []) + [f"orch->{nxt}"]}

def route_orchestrator(state: ChatState) -> str:
    return (state.get("eval", {}) or {}).get("next", "final")

# ===================== Graph wiring =====================
state_graph = StateGraph(ChatState)
state_graph.add_node("router",        router_node)
state_graph.add_node("orchestrator",  orchestrator_node)
state_graph.add_node("query_rewriter", query_rewriter_node)
state_graph.add_node("retrieve",      retriever_node)
state_graph.add_node("rerank",        reranker_node)
state_graph.add_node("react_loop",    react_loop_node)
state_graph.add_node("self_ask_loop", self_ask_loop_node)
state_graph.add_node("llm",           llm_node)

state_graph.add_edge(START, "router")
state_graph.add_edge("router", "orchestrator")

state_graph.add_conditional_edges("orchestrator", route_orchestrator, {
    "retrieve": "retrieve",
    "react": "react_loop",
    "self_ask": "self_ask_loop",
    "final": "llm",
})

state_graph.add_edge("retrieve", "rerank")
state_graph.add_edge("rerank", "orchestrator")
state_graph.add_conditional_edges("react_loop", route_react_loop, {"loop": "react_loop", "done": END})
state_graph.add_conditional_edges("self_ask_loop", route_selfask_loop, {"loop": "self_ask_loop", "done": END})
state_graph.add_edge("llm", END)

# Export compiled graph for Studio
graph = build_graph()

