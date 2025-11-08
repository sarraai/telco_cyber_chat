import os
import re
import numpy as np
from functools import lru_cache
from typing import List
from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------------
USE_REMOTE = os.getenv("USE_REMOTE_HF", "true").lower() == "true"

# HF router (OpenAI-compatible) base
REMOTE_BASE = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")

# Pin both generator and guard to featherless-ai by default
REMOTE_MODEL_ID = os.getenv("REMOTE_MODEL_ID", "fdtn-ai/Foundation-Sec-8B-Instruct:featherless-ai")
REMOTE_GUARD_ID = os.getenv("REMOTE_GUARD_ID", "meta-llama/Llama-Guard-3-8B:featherless-ai")

# Force a specific provider everywhere unless overridden
HF_PROVIDER = os.getenv("HF_PROVIDER", "featherless-ai")
# Keep guard on the same provider as generator (true by default)
HF_ALIGN_GUARD_PROVIDER = os.getenv("HF_ALIGN_GUARD_PROVIDER", "true").lower() == "true"
# Optional: allow streaming; we'll buffer tokens so callers still get a string
HF_STREAM = os.getenv("HF_STREAM", "false").lower() == "true"

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("LLM_TOKEN")

# -----------------------------------------------------------------------------
# Small-talk guard (prevents canned telco bullets on greetings/thanks)
# -----------------------------------------------------------------------------
GREETING_REPLY = os.getenv("GREETING_REPLY", "Hello! I'm your telecom-cybersecurity assistant.")
GOODBYE_REPLY  = os.getenv("GOODBYE_REPLY", "Goodbye!")

# FIXED: More precise regex that matches ENTIRE string (not just start)
_GREET_RE = re.compile(r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day))\s*[!.?]*\s*$", re.I)
_BYE_RE   = re.compile(r"^\s*(bye|goodbye|see\s+you|see\s+ya|thanks?\s*,?\s*bye|farewell)\s*[!.?]*\s*$", re.I)
_THANKS_RE = re.compile(r"^\s*(thanks|thank\s+you|thx)\s*[!.?]*\s*$", re.I)

def _smalltalk_reply(text: str):
    """
    CRITICAL: This must be called FIRST before any LLM processing.
    Returns immediate reply for greetings/goodbyes, None otherwise.
    """
    t = (text or "").strip()
    if _GREET_RE.search(t):
        return GREETING_REPLY
    if _BYE_RE.search(t):
        return GOODBYE_REPLY
    if _THANKS_RE.search(t):
        return "You're welcome!"
    # Extremely short non-question → treat as small talk
    if len(re.findall(r"\w+", t)) <= 2 and not t.endswith("?"):
        return GREETING_REPLY
    return None

# -----------------------------------------------------------------------------
# BGE similarity via HF Inference (no local weights, no downloads on import)
# -----------------------------------------------------------------------------
BGE_MODEL_ID = os.getenv("BGE_MODEL_ID", "BAAI/bge-m3")

@lru_cache(maxsize=1)
def _get_bge_client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN env var is missing")
    # Let InferenceClient pick the right HF endpoints; don't hardcode /hf-inference
    return InferenceClient(api_key=HF_TOKEN)

def bge_sentence_similarity(source: str, candidates: List[str], model: str = BGE_MODEL_ID) -> List[float]:
    """
    Returns cosine-similarity scores between `source` and each text in `candidates`.
    Tries HF's sentence_similarity task first; falls back to feature_extraction + cosine.
    """
    client = _get_bge_client()
    try:
        return client.sentence_similarity(
            {"source_sentence": source, "sentences": candidates},
            model=model,
        )
    except Exception:
        # Fallback: embed + cosine similarity
        def l2(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
        s = np.array(client.feature_extraction(source, model=model))
        C = np.array(client.feature_extraction(candidates, model=model))
        if s.ndim == 1:
            s = s[0][None, :] if s.size and isinstance(s[0], (list, np.ndarray)) else s[None, :]
        if C.ndim == 1:
            C = C[None, :]
        s, C = l2(s)[0], l2(C)
        return (C @ s).tolist()

# -----------------------------------------------------------------------------
# Remote (HF Router via OpenAI-compatible API)
# -----------------------------------------------------------------------------
if USE_REMOTE:
    from openai import OpenAI

    _client = None

    def _normalize_base(url: str) -> str:
        base = (url or "").rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

    def _with_provider(mid: str, provider: str) -> str:
        """Attach/replace provider suffix '<repo>:<provider>'."""
        return f"{(mid or '').split(':', 1)[0]}:{provider}"

    def get_client():
        global _client
        if _client is None:
            if not HF_TOKEN:
                raise RuntimeError("HF_TOKEN env var is missing")
            _client = OpenAI(base_url=_normalize_base(REMOTE_BASE), api_key=HF_TOKEN)
        return _client

    @lru_cache(maxsize=1)
    def _get_gen_model_id():
        # Force generator to HF_PROVIDER (default featherless-ai)
        return _with_provider(REMOTE_MODEL_ID, HF_PROVIDER)

    @lru_cache(maxsize=1)
    def _get_guard_model_id():
        # Keep guard on the same provider unless explicitly disabled
        if HF_ALIGN_GUARD_PROVIDER:
            return _with_provider(REMOTE_GUARD_ID, HF_PROVIDER)
        # Otherwise ensure *some* provider is present
        if ":" not in (REMOTE_GUARD_ID or ""):
            return _with_provider(REMOTE_GUARD_ID, HF_PROVIDER)
        return REMOTE_GUARD_ID

    def generate_text(prompt: str, **decoding) -> str:
        client = get_client()
        max_tokens = int(decoding.get("max_new_tokens", 512))
        temperature = float(decoding.get("temperature", 0.0))
        top_p = float(decoding.get("top_p", 1.0))

        resp = client.chat.completions.create(
            model=_get_gen_model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=HF_STREAM,
        )

        if not HF_STREAM:
            return (resp.choices[0].message.content or "").strip()

        # Streamed mode: buffer tokens so callers still get a single string
        parts = []
        for ev in resp:
            try:
                delta = ev.choices[0].delta.content
                if delta:
                    parts.append(delta)
            except Exception:
                pass
        return "".join(parts).strip()

    # ---------- LlamaGuard + role-adaptive prompt ----------
    POLICY = {
        "end_user": {"deny": {"S1 Violent","S2 Non-violent Crimes","S3 Sex-crime","S4 CSE",
                              "S7 Privacy","S9 Indiscriminate Weapons","S10 Hate",
                              "S11 Self-Harm","S12 Sexual","S13 Elections","Malicious Code","Phishing/SE"},
                     "allow_defense_only": set()},
        "it_specialist": {"deny": {"S7 Privacy","S9 Indiscriminate Weapons","S10 Hate","S11 Self-Harm","S12 Sexual"},
                          "allow_defense_only": {"Malicious Code","Phishing/SE","S2 Non-violent Crimes","S6 Specialized Illicit Advice"}},
        "network_admin": {"deny": {"S7 Privacy","S9 Indiscriminate Weapons","S10 Hate","S11 Self-Harm","S12 Sexual"},
                          "allow_defense_only": {"Malicious Code","Phishing/SE","S2 Non-violent Crimes","S6 Specialized Illicit Advice"}},
        "admin": {"deny": {"S7 Privacy","S9 Indiscriminate Weapons","S10 Hate","S11 Self-Harm","S12 Sexual","S4 CSE"},
                  "allow_defense_only": {"Malicious Code","Phishing/SE","S2 Non-violent Crimes","S6 Specialized Illicit Advice"}},
    }
    HAZARDS = ("S1 Violent; S2 Non-violent Crimes; S3 Sex-crime; S4 CSE; S5 Defamation; "
               "S6 Specialized Illicit Advice; S7 Privacy; S8 IP; S9 Indiscriminate Weapons; "
               "S10 Hate; S11 Self-Harm; S12 Sexual; S13 Elections; S14 Code Interpreter Abuse; "
               "Malicious Code; Phishing/SE")
    FORMAT_HELP = ("\nRespond EXACTLY in this format:\n"
                   "DECISION: SAFE or UNSAFE\nCATEGORIES: [comma-separated subset from the list above]\n")
    PROMPT_IN  = ("Classify the USER INPUT as SAFE or UNSAFE; list categories from: {haz}." + FORMAT_HELP +
                  "USER INPUT:\n{text}\n\nDECISION:")
    PROMPT_OUT = ("Classify the MODEL RESPONSE as SAFE or UNSAFE; list categories from: {haz}." + FORMAT_HELP +
                  "MODEL RESPONSE:\n{text}\n\nDECISION:")
    PATTERN = re.compile(r"DECISION:\s*(SAFE|UNSAFE).*?CATEGORIES:\s*\[([^\]]*)\]", re.I | re.S)

    def _canon_role(role: str) -> str:
        r = (role or "").strip().lower().replace(" ", "_")
        return r if r in POLICY else "end_user"

    def _classify_with_guard(text: str, mode: str):
        client = get_client()
        prompt = (PROMPT_IN if mode == "input" else PROMPT_OUT).format(haz=HAZARDS, text=text)
        r = client.chat.completions.create(
            model=_get_guard_model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, top_p=1.0, max_tokens=128, stream=False,
        )
        out = r.choices[0].message.content or ""
        m = PATTERN.search(out)
        decision = (m.group(1).upper() if m else "SAFE")
        cats = set(c.strip() for c in (m.group(2).split(",") if m else []) if c.strip())
        return decision, cats

    def guard_pre(user_text: str, role: str = "end_user"):
        role = _canon_role(role)
        decision, cats = _classify_with_guard(user_text, "input")
        pol = POLICY[role]
        if decision == "UNSAFE" and (cats & pol["deny"]):
            return False, {"why": "blocked_input", "categories": sorted(cats), "role": role}
        return True, {"defense_only": bool(cats & pol["allow_defense_only"]), "categories": sorted(cats), "role": role}

    def guard_post(model_text: str, role: str = "end_user"):
        role = _canon_role(role)
        decision, cats = _classify_with_guard(model_text, "output")
        pol = POLICY[role]
        if decision == "UNSAFE" and (cats & pol["deny"]):
            return False, {"why": "blocked_output", "categories": sorted(cats), "role": role}
        return True, {"categories": sorted(cats), "role": role}

    def role_directive(role:str)->str:
        r = (role or "").lower().replace(" ", "_")
        if r == "end_user":
            return "Audience: non-technical end user. Explain simply, avoid jargon.\nOutput: ≤6 short bullets.\n"
        if r == "it_specialist":
            return "Audience: IT specialist. Provide technical bullets + brief rationale.\n"
        if r == "network_admin":
            return "Audience: network admin. Focus on configs, controls, rollout steps.\n"
        if r == "admin":
            return "Audience: executive/admin. 5-line summary: risk, impact, priority, owners.\n"
        return "Audience: general user. Be concise.\n"

    def build_prompt(question:str, context:str, *, role:str="end_user", defense_only:bool=False)->str:
        """
        FIXED: Don't force context-only mode for greetings/small-talk.
        Allow natural responses when context is empty or irrelevant.
        """
        # If no context provided, allow freeform response
        if not context.strip():
            return (f"{role_directive(role)}"
                    "You are a telecom-cybersecurity assistant.\n"
                    "Answer the question naturally and concisely.\n\n"
                    f"Question:\n{question.strip()}\n\nAnswer:")
        
        # Normal RAG mode with context
        safety = ("Provide defensive mitigations only. Do NOT include exploit code, payloads, or targeting steps.\n"
                  if defense_only else "")
        
        return (f"{role_directive(role)}{safety}"
                "You are a telecom-cybersecurity assistant.\n"
                "- Use the Context when relevant to answer the question.\n"
                "- For greetings or small-talk, respond naturally without forcing technical content.\n"
                "- If the question requires specific info not in context, say: 'Not enough evidence in context.'\n"
                "- Cite snippets with [D#]. No chain-of-thought. No sensitive data.\n\n"
                f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}\n\nAnswer:")

    SAMPLING_PRESETS = {
        "factual":  {"temperature": 0.3, "top_p": 0.9},
        "balanced": {"temperature": 0.7, "top_p": 0.9},
        "creative": {"temperature": 1.2, "top_p": 0.92},
    }

    def refusal_message() -> str:
        return ("I can't help with that because it's outside my allowed use. "
                "Here's a safe alternative: high-level risks, mitigations, and references.\n")

    def ask_secure(question: str, *, context: str = "", role: str = "end_user",
                   max_new_tokens: int = 400, preset: str = "balanced", seed: int | None = None) -> str:
        """
        CRITICAL FIX: Small-talk check MUST happen FIRST, before any guards or LLM calls.
        """
        # ---- Small-talk fast path (short-circuit) ----
        st = _smalltalk_reply(question)
        if st is not None:
            return st

        # ---- Normal RAG flow ----
        client = get_client()
        ok, info = guard_pre(question, role=role)
        if not ok:
            return refusal_message()
        
        prompt = build_prompt(question, context, role=role, defense_only=info.get("defense_only", False))
        preset_vals = SAMPLING_PRESETS.get(preset, SAMPLING_PRESETS["balanced"])
        
        r = client.chat.completions.create(
            model=_get_gen_model_id(),
            messages=[{"role": "user", "content": prompt}],
            temperature=float(preset_vals["temperature"]),
            top_p=float(preset_vals["top_p"]),
            max_tokens=int(max_new_tokens),
            stream=False,
        )
        out = (r.choices[0].message.content or "").strip()
        ok2, _ = guard_post(out, role=role)
        return out if ok2 else refusal_message()

# -----------------------------------------------------------------------------
# Local transformers fallback (when USE_REMOTE_HF != true)
# -----------------------------------------------------------------------------
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
    import torch

    def _load(repo_id: str, prefer_4bit: bool = True):
        tok_kw = {"token": HF_TOKEN} if HF_TOKEN else {}
        tok = AutoTokenizer.from_pretrained(repo_id, use_fast=True, **tok_kw)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        use_gpu = torch.cuda.is_available()
        bf16_ok = use_gpu and torch.cuda.get_device_capability(0)[0] >= 8
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        mdl = None
        if use_gpu and prefer_4bit:
            try:
                q4 = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype
                )
                mdl = AutoModelForCausalLM.from_pretrained(
                    repo_id, device_map="auto", quantization_config=q4,
                    low_cpu_mem_usage=True, trust_remote_code=False, **tok_kw
                )
            except Exception:
                mdl = None
        if mdl is None and use_gpu:
            try:
                q8 = BitsAndBytesConfig(load_in_8bit=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    repo_id, device_map="auto", quantization_config=q8,
                    low_cpu_mem_usage=True, trust_remote_code=False, **tok_kw
                )
            except Exception:
                mdl = None
        if mdl is None and use_gpu:
            try:
                mdl = AutoModelForCausalLM.from_pretrained(
                    repo_id, device_map="auto", dtype="auto",
                    low_cpu_mem_usage=True, trust_remote_code=False, **tok_kw
                )
            except Exception:
                mdl = None
        if mdl is None:
            mdl = AutoModelForCausalLM.from_pretrained(
                repo_id, device_map={"": "cpu"},
                low_cpu_mem_usage=True, trust_remote_code=False, **tok_kw
            )
        mdl.eval()
        return tok, mdl

    @lru_cache(maxsize=1)
    def _get_gen():
        tok, mdl = _load(os.getenv("MODEL_ID", "fdtn-ai/Foundation-Sec-8B-Instruct"), prefer_4bit=True)
        return pipeline(
            "text-generation", model=mdl, tokenizer=tok,
            do_sample=False, num_beams=1, top_p=1.0,
            max_new_tokens=512, return_full_text=False, pad_token_id=tok.pad_token_id
        )

    @lru_cache(maxsize=1)
    def _get_guard():
        tok, mdl = _load(os.getenv("GUARD_ID", "meta-llama/Llama-Guard-3-8B"), prefer_4bit=True)
        return pipeline(
            "text-generation", model=mdl, tokenizer=tok,
            do_sample=False, num_beams=1, top_p=1.0,
            max_new_tokens=128, return_full_text=True, pad_token_id=tok.pad_token_id
        )

    def generate_text(prompt: str, **decoding) -> str:
        gen = _get_gen()
        decoding = {"do_sample": False, "num_beams": 1, "top_p": 1.0, **(decoding or {})}
        return gen(prompt, **decoding)[0]["generated_text"].strip()

    # ---------- LlamaGuard + role-adaptive prompt (local pipeline) ----------
    POLICY = {
        "end_user": {"deny": {"S1 Violent","S2 Non-violent Crimes","S3 Sex-crime","S4 CSE",
                              "S7 Privacy","S9 Indiscriminate Weapons","S10 Hate",
                              "S11 Self-Harm","S12 Sexual","S13 Elections","Malicious Code","Phishing/SE"},
                     "allow_defense_only": set()},
        "it_specialist": {"deny": {"S7 Privacy","S9 Indiscriminate Weapons","S10 Hate","S11 Self-Harm","S12 Sexual"},
                          "allow_defense_only": {"Malicious Code","Phishing/SE","S2 Non-violent Crimes","S6 Specialized Illicit Advice"}},
        "network_admin": {"deny": {"S7 Privacy","S9 Indiscriminate Weapons","S10 Hate","S11 Self-Harm","S12 Sexual"},
                          "allow_defense_only": {"Malicious Code","Phishing/SE","S2 Non-violent Crimes","S6 Specialized Illicit Advice"}},
        "admin": {"deny": {"S7 Privacy","S9 Indiscriminate Weapons","S10 Hate","S11 Self-Harm","S12 Sexual","S4 CSE"},
                  "allow_defense_only": {"Malicious Code","Phishing/SE","S2 Non-violent Crimes","S6 Specialized Illicit Advice"}},
    }
    HAZARDS = ("S1 Violent; S2 Non-violent Crimes; S3 Sex-crime; S4 CSE; S5 Defamation; "
               "S6 Specialized Illicit Advice; S7 Privacy; S8 IP; S9 Indiscriminate Weapons; "
               "S10 Hate; S11 Self-Harm; S12 Sexual; S13 Elections; S14 Code Interpreter Abuse; "
               "Malicious Code; Phishing/SE")
    FORMAT_HELP = ("\nRespond EXACTLY in this format:\n"
                   "DECISION: SAFE or UNSAFE\nCATEGORIES: [comma-separated subset from the list above]\n")
    PROMPT_IN  = ("Classify the USER INPUT as SAFE or UNSAFE; list categories from: {haz}." + FORMAT_HELP +
                  "USER INPUT:\n{text}\n\nDECISION:")
    PROMPT_OUT = ("Classify the MODEL RESPONSE as SAFE or UNSAFE; list categories from: {haz}." + FORMAT_HELP +
                  "MODEL RESPONSE:\n{text}\n\nDECISION:")
    PATTERN = re.compile(r"DECISION:\s*(SAFE|UNSAFE).*?CATEGORIES:\s*\[([^\]]*)\]", re.I | re.S)

    def _canon_role(role: str) -> str:
        r = (role or "").strip().lower().replace(" ", "_")
        return r if r in POLICY else "end_user"

    def _classify_with_guard(text: str, mode: str):
        guard = _get_guard()
        prompt = (PROMPT_IN if mode == "input" else PROMPT_OUT).format(haz=HAZARDS, text=text)
        out = guard(prompt)[0]["generated_text"]
        m = PATTERN.search(out or "")
        decision = (m.group(1).upper() if m else "SAFE")
        cats = set(c.strip() for c in (m.group(2).split(",") if m else []) if c.strip())
        return decision, cats

    def guard_pre(user_text: str, role: str = "end_user"):
        role = _canon_role(role)
        decision, cats = _classify_with_guard(user_text, "input")
        pol = POLICY[role]
        if decision == "UNSAFE" and (cats & pol["deny"]):
            return False, {"why": "blocked_input", "categories": sorted(cats), "role": role}
        return True, {"defense_only": bool(cats & pol["allow_defense_only"]), "categories": sorted(cats), "role": role}

    def guard_post(model_text: str, role: str = "end_user"):
        role = _canon_role(role)
        decision, cats = _classify_with_guard(model_text, "output")
        pol = POLICY[role]
        if decision == "UNSAFE" and (cats & pol["deny"]):
            return False, {"why": "blocked_output", "categories": sorted(cats), "role": role}
        return True, {"categories": sorted(cats), "role": role}

    def role_directive(role:str)->str:
        r = (role or "").lower().replace(" ", "_")
        if r == "end_user":
            return "Audience: non-technical end user. Explain simply, avoid jargon.\nOutput: ≤6 short bullets.\n"
        if r == "it_specialist":
            return "Audience: IT specialist. Provide technical bullets + brief rationale.\n"
        if r == "network_admin":
            return "Audience: network admin. Focus on configs, controls, rollout steps.\n"
        if r == "admin":
            return "Audience: executive/admin. 5-line summary: risk, impact, priority, owners.\n"
        return "Audience: general user. Be concise.\n"

    def build_prompt(question:str, context:str, *, role:str="end_user", defense_only:bool=False)->str:
        """
        FIXED: Allow natural responses when context is empty (greetings, etc.)
        """
        # If no context, allow freeform response
        if not context.strip():
            return (f"{role_directive(role)}"
                    "You are a telecom-cybersecurity assistant.\n"
                    "Answer the question naturally and concisely.\n\n"
                    f"Question:\n{question.strip()}\n\nAnswer:")
        
        # Normal RAG mode
        safety = ("Provide defensive mitigations only. Do NOT include exploit code, payloads, or targeting steps.\n"
                  if defense_only else "")
        
        return (f"{role_directive(role)}{safety}"
                "You are a telecom-cybersecurity assistant.\n"
                "- Answer using BOTH the provided context AND your cybersecurity knowledge.\n"
                "- For greetings or small-talk, respond naturally.\n\n"
                f"Context from Database:\n{context.strip()}\n\n"
                f"Question:\n{question.strip()}\n\n"
                "Answer:")

    SAMPLING_PRESETS = {
        "factual":  {"temperature": 0.3, "top_p": 0.9},
        "balanced": {"temperature": 0.7, "top_p": 0.9},
        "creative": {"temperature": 1.2, "top_p": 0.92},
    }

    def refusal_message() -> str:
        return ("I can't help with that because it's outside my allowed use. "
                "Here's a safe alternative: high-level risks, mitigations, and references.\n")

    def ask_secure(question: str, *, context: str = "", role: str = "end_user",
                   max_new_tokens: int = 400, preset: str = "balanced", seed: int | None = None) -> str:
        """
        CRITICAL FIX: Small-talk MUST be checked FIRST.
        """
        # ---- Small-talk fast path (short-circuit) ----
        st = _smalltalk_reply(question)
        if st is not None:
            return st

        ok, info = guard_pre(question, role=role)
        if not ok:
            return refusal_message()
        prompt = build_prompt(question, context, role=role, defense_only=info.get("defense_only", False))
        preset_vals = SAMPLING_PRESETS.get(preset, SAMPLING_PRESETS["balanced"])
        gen = _get_gen()
        out = gen(
            prompt,
            do_sample=False,
            num_beams=1,
            top_p=float(preset_vals["top_p"]),
            max_new_tokens=int(max_new_tokens)
        )[0]["generated_text"].strip()
        ok2, _ = guard_post(out, role=role)
        return out if ok2 else refusal_message()
