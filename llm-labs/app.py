"""
Laboratorio de LLMs con Groq: modelos, tokens e IDs, bag of words, métricas de similitud,
embeddings y generación de texto con parámetros ajustables.

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py

Notas:
- Hasta donde sé, Groq no ofrece endpoint de embeddings; por eso los embeddings se calculan
  localmente con sentence-transformers (la primera vez descarga el modelo de Hugging Face).
- tiktoken descarga sus vocabularios la primera vez que se usa cada codificación.
"""
from __future__ import annotations

import html
import os
import re
import time
from difflib import SequenceMatcher
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
DEFAULT_CORPUS = """El gato duerme sobre el sofá de la sala.
Un felino descansa encima del sillón de la casa.
Los modelos de lenguaje predicen el siguiente token de una secuencia.
Los transformers usan atención para procesar secuencias de tokens.
La bolsa de palabras ignora el orden y cuenta frecuencias."""

TOKEN_RE = r"(?u)\b\w+\b"

STOP_ES = sorted(set(
    """de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí
    porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno
    les ni contra otros ese eso ante ellos e esto antes algunos qué unos yo otro otras otra él tanto esa estos
    mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros mi mis tú te ti tu tus ellas
    son es ser fue ha han era está están""".split()
))

TIKTOKEN_ENCODINGS = {
    "o200k_harmony": "GPT-OSS (formato Harmony)",
    "o200k_base": "GPT-4o / serie o",
    "cl100k_base": "GPT-4 / GPT-3.5",
    "p50k_base": "Codex / text-davinci",
    "r50k_base": "GPT-3 (davinci)",
}
HF_TOKENIZERS = ["openai-community/gpt2", "Qwen/Qwen2.5-0.5B", "openai/gpt-oss-20b"]

EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

REFERENCE_GPT = pd.DataFrame({
    "id": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"],
    "descripción": ["GPT-OSS 120B (razonamiento)", "GPT-OSS 20B (razonamiento)", "GPT-OSS Safeguard 20B (clasificación de políticas)"],
})

PALETTE = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fbcfe8"]

SIM_METRICS = [
    "Coseno (BoW)", "Coseno (TF-IDF)", "Jaccard (palabras)", "Euclidiana (BoW)",
    "Manhattan (BoW)", "Secuencia (difflib, caracteres)", "Coseno (embeddings)",
]


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def parse_corpus(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def is_chat_model(mid: str) -> bool:
    bad = ("whisper", "tts", "orpheus", "playai", "prompt-guard", "llama-guard", "embed")
    return not any(b in mid.lower() for b in bad)


def list_groq_models(api_key: str) -> list[dict]:
    from groq import Groq

    data = Groq(api_key=api_key).models.list().data
    return [{"id": m.id, "propietario": getattr(m, "owned_by", ""),
             "contexto": getattr(m, "context_window", None), "activo": getattr(m, "active", None)}
            for m in data]


# ----------------------------------------------------------------------------
# Tokenización
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando tokenizer...")
def get_tokenizer(kind: str, name: str):
    if kind == "tiktoken":
        import tiktoken

        return tiktoken.get_encoding(name)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def tokenize(kind: str, name: str, text: str) -> tuple[list[int], list[str]]:
    tok = get_tokenizer(kind, name)
    if kind == "tiktoken":
        ids = tok.encode(text, disallowed_special=())
        pieces = [tok.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace") for i in ids]
    else:
        ids = tok.encode(text, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
    return list(ids), list(pieces)


def decode_ids(kind: str, name: str, ids: list[int]) -> str:
    return get_tokenizer(kind, name).decode(ids)


def visible(p: str) -> str:
    return p.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥")


def colored_tokens(pieces: list[str]) -> str:
    spans = [
        f'<span style="background:{PALETTE[i % len(PALETTE)]};color:#111;padding:2px 1px;border-radius:3px;'
        f'margin:1px;display:inline-block;font-family:monospace">{html.escape(visible(p))}</span>'
        for i, p in enumerate(pieces)
    ]
    return "".join(spans)


# ----------------------------------------------------------------------------
# Embeddings (locales) y similitud
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de embeddings (la primera vez se descarga)...")
def get_embedder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@st.cache_data(show_spinner=False)
def embed_cached(texts: tuple, name: str, normalize: bool) -> np.ndarray:
    return np.asarray(get_embedder(name).encode(list(texts), normalize_embeddings=normalize,
                                                show_progress_bar=False))


def jaccard_matrix(docs: list[str]) -> np.ndarray:
    sets = [set(re.findall(r"\w+", d.lower())) for d in docs]
    n = len(docs)
    M = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            u = len(sets[i] | sets[j])
            M[i, j] = len(sets[i] & sets[j]) / u if u else 1.0
    return M


def seq_matrix(docs: list[str]) -> np.ndarray:
    n = len(docs)
    return np.array([[SequenceMatcher(None, docs[i].lower(), docs[j].lower()).ratio() for j in range(n)]
                     for i in range(n)])


def compute_matrix(metric: str, docs: list[str], stop: list[str] | None, emb_name: str):
    """Devuelve (matriz, 'similitud' | 'distancia')."""
    if metric in ("Coseno (BoW)", "Euclidiana (BoW)", "Manhattan (BoW)"):
        X = CountVectorizer(token_pattern=TOKEN_RE, stop_words=stop).fit_transform(docs)
        if metric == "Coseno (BoW)":
            return cosine_similarity(X), "similitud"
        if metric == "Euclidiana (BoW)":
            return euclidean_distances(X), "distancia"
        return manhattan_distances(X), "distancia"
    if metric == "Coseno (TF-IDF)":
        return cosine_similarity(TfidfVectorizer(token_pattern=TOKEN_RE, stop_words=stop).fit_transform(docs)), "similitud"
    if metric == "Jaccard (palabras)":
        return jaccard_matrix(docs), "similitud"
    if metric == "Secuencia (difflib, caracteres)":
        return seq_matrix(docs), "similitud"
    E = embed_cached(tuple(docs), emb_name, True)
    return cosine_similarity(E), "similitud"


# ----------------------------------------------------------------------------
# Generación con Groq
# ----------------------------------------------------------------------------
def groq_generate(api_key: str, model: str, messages: list[dict], params: dict) -> dict:
    """Llama a Groq. Si el modelo rechaza algún parámetro, lo omite y reintenta."""
    from groq import BadRequestError, Groq

    client = Groq(api_key=api_key)
    p = {k: v for k, v in params.items() if v is not None}
    dropped: list[str] = []
    last: Exception | None = None
    for _ in range(8):
        kw: dict = {"model": model, "messages": messages}
        extra: dict = {}
        for k, v in p.items():
            (extra if k in ("reasoning_effort", "include_reasoning") else kw)[k] = v
        if extra:
            kw["extra_body"] = extra
        try:
            t0 = time.perf_counter()
            r = client.chat.completions.create(**kw)
            secs = time.perf_counter() - t0
            msg = r.choices[0].message
            u = r.usage
            comp_time = getattr(u, "completion_time", None)
            return {
                "text": msg.content or "", "reasoning": getattr(msg, "reasoning", None),
                "finish": r.choices[0].finish_reason, "seconds": secs, "dropped": dropped,
                "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
                "tps": (u.completion_tokens / comp_time) if comp_time else (u.completion_tokens / secs if secs else None),
            }
        except BadRequestError as e:
            last = e
            param = (getattr(e, "param", None) or "").split(".")[0]
            if not param:
                m = re.search(r"'(\w+)'", str(e))
                param = m.group(1) if m else ""
            if param in p:
                dropped.append(param)
                p.pop(param)
                continue
            raise
    raise last  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
def sidebar() -> SimpleNamespace:
    with st.sidebar:
        st.header("🔑 Groq")
        key = st.text_input("API key de Groq", type="password", value=os.getenv("GROQ_API_KEY", ""),
                            placeholder="gsk_...", help="Solo vive en esta sesión; no se guarda en disco.")
        if st.button("Conectar y listar modelos", type="primary"):
            try:
                st.session_state["models"] = list_groq_models(key.strip())
                st.session_state["api_key"] = key.strip()
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("models", None)
                st.error(f"No se pudo conectar: {e}")
        only_gpt = st.checkbox("Solo modelos GPT (openai/…)", True)
        models = st.session_state.get("models")
        if models:
            st.success(f"Conectado · {len(models)} modelos")
        else:
            st.caption("Ingresa tu API key y pulsa «Conectar». Las pestañas de tokens, BoW, similitud y "
                       "embeddings funcionan sin API key.")
    return SimpleNamespace(api_key=st.session_state.get("api_key", ""), models=models or [], only_gpt=only_gpt)


def chat_options(cfg: SimpleNamespace) -> list[str]:
    ids = [m["id"] for m in cfg.models if is_chat_model(m["id"])]
    gpt = [i for i in ids if i.startswith("openai/")]
    opts = gpt if (cfg.only_gpt and gpt) else ids
    return sorted(opts, key=lambda s: (not s.startswith("openai/"), s))


def tab_models(cfg: SimpleNamespace) -> None:
    if not cfg.models:
        st.info("Conecta tu API key para ver los modelos disponibles en tu cuenta.")
        st.markdown("**Referencia: modelos GPT en Groq** (puede cambiar; la lista real sale al conectar)")
        st.dataframe(REFERENCE_GPT, hide_index=True)
        return
    df = pd.DataFrame(cfg.models)
    df["GPT (OpenAI)"] = df["id"].str.startswith("openai/")
    df["tipo"] = np.where(df["id"].map(is_chat_model), "chat", "otro (audio/guard/...)")
    c = st.columns(3)
    c[0].metric("Modelos totales", len(df))
    c[1].metric("Modelos GPT", int(df["GPT (OpenAI)"].sum()))
    c[2].metric("Modelos de chat", int((df["tipo"] == "chat").sum()))
    view = df[df["GPT (OpenAI)"]] if cfg.only_gpt and df["GPT (OpenAI)"].any() else df
    st.dataframe(view.sort_values(["GPT (OpenAI)", "id"], ascending=[False, True]), hide_index=True)
    chart = df[(df["tipo"] == "chat") & df["contexto"].notna()].sort_values("contexto")
    if not chart.empty:
        st.plotly_chart(px.bar(chart, x="contexto", y="id", orientation="h", color="GPT (OpenAI)",
                               title="Ventana de contexto (tokens)"))
    st.caption("Los modelos openai/gpt-oss-* son de razonamiento: en «Generación» puedes ajustar `reasoning_effort`.")


def tab_tokens() -> None:
    text = st.text_area("Texto a tokenizar", "Los modelos de lenguaje convierten el texto en tokens e IDs. ¡Hola, mundo! 🌎",
                        height=100)
    labels = [f"tiktoken · {k}  ({v})" for k, v in TIKTOKEN_ENCODINGS.items()] + [f"HF · {n}" for n in HF_TOKENIZERS]
    sel = st.multiselect("Tokenizers", labels, default=labels[:3])
    custom = st.text_input("Tokenizer de Hugging Face personalizado (opcional)", placeholder="p. ej. Qwen/Qwen2.5-0.5B")
    specs: list[tuple[str, str, str]] = []
    for lab in sel:
        if lab.startswith("tiktoken"):
            name = lab.split("·")[1].split()[0]
            specs.append((lab, "tiktoken", name))
        else:
            specs.append((lab, "hf", lab.split("· ")[1]))
    if custom.strip():
        specs.append((f"HF · {custom.strip()}", "hf", custom.strip()))
    if not text.strip() or not specs:
        st.info("Escribe un texto y elige al menos un tokenizer.")
        return

    rows, results = [], {}
    for label, kind, name in specs:
        try:
            ids, pieces = tokenize(kind, name, text)
            results[label] = (kind, name, ids, pieces)
            rows.append({"tokenizer": name, "tokens": len(ids), "caracteres": len(text),
                         "caracteres/token": round(len(text) / max(len(ids), 1), 2),
                         "bytes UTF-8": len(text.encode("utf-8"))})
        except Exception as e:  # noqa: BLE001
            st.error(f"{name}: {type(e).__name__}: {e}")
    if not rows:
        return
    summary = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 1])
    c1.dataframe(summary, hide_index=True)
    c2.plotly_chart(px.bar(summary, x="tokenizer", y="tokens", title="Tokens por tokenizer", text="tokens"))

    for i, (label, (kind, name, ids, pieces)) in enumerate(results.items()):
        with st.expander(f"{name} — {len(ids)} tokens", expanded=(i == 0)):
            st.markdown(colored_tokens(pieces), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"pos": range(len(ids)), "token": [visible(p) for p in pieces], "id": ids}),
                         hide_index=True)
            st.code(str(ids), language="text")

    st.subheader("Decodificar IDs → texto")
    d1, d2 = st.columns([2, 1])
    ids_txt = d1.text_input("IDs separados por coma", placeholder="p. ej. 12, 345, 6789")
    names = list(results)
    which = d2.selectbox("Con el tokenizer", names, key="dec_tok")
    if ids_txt.strip():
        try:
            ids = [int(x) for x in re.split(r"[,\s]+", ids_txt.strip()) if x]
            kind, name, _, _ = results[which]
            st.success(repr(decode_ids(kind, name, ids)))
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo decodificar: {e}")


def tab_bow(docs: list[str]) -> None:
    if not docs:
        st.warning("El corpus está vacío.")
        return
    c1, c2, c3, c4 = st.columns(4)
    lowercase = c1.checkbox("Minúsculas", True)
    binary = c1.checkbox("Binario (presencia)", False)
    ngram = c2.slider("n-gramas", 1, 3, (1, 1))
    min_df = c2.number_input("min_df", 1, 20, 1)
    max_feat = c3.number_input("max_features (0 = todas)", 0, 5000, 0)
    use_stop = c3.checkbox("Quitar stopwords (es)", False)
    weight = c4.radio("Ponderación", ["Conteo", "TF-IDF"])
    kw = dict(lowercase=lowercase, ngram_range=ngram, min_df=int(min_df), token_pattern=TOKEN_RE,
              max_features=int(max_feat) or None, stop_words=STOP_ES if use_stop else None)
    try:
        vec = CountVectorizer(binary=binary, **kw) if weight == "Conteo" else TfidfVectorizer(binary=binary, **kw)
        X = vec.fit_transform(docs)
    except ValueError as e:
        st.error(f"No se pudo vectorizar: {e}")
        return
    terms = vec.get_feature_names_out()
    df = pd.DataFrame(X.toarray(), index=[f"D{i + 1}" for i in range(len(docs))], columns=terms)
    m = st.columns(3)
    m[0].metric("Vocabulario", len(terms))
    m[1].metric("Forma de la matriz", f"{X.shape[0]} × {X.shape[1]}")
    m[2].metric("Densidad", f"{100 * X.nnz / (X.shape[0] * X.shape[1]):.1f} %")
    st.dataframe(df.round(3))
    top = df.sum().sort_values(ascending=False).head(15)
    g1, g2 = st.columns(2)
    g1.plotly_chart(px.bar(top[::-1], orientation="h", title="Términos más pesados", labels={"value": "suma", "index": ""}))
    cols = list(df.sum().sort_values(ascending=False).head(30).index)
    g2.plotly_chart(px.imshow(df[cols], aspect="auto", color_continuous_scale="Blues",
                              title="Mapa de calor (30 términos principales)"))


def tab_sim(docs: list[str]) -> None:
    if len(docs) < 2:
        st.warning("Necesitas al menos 2 documentos en el corpus.")
        return
    c1, c2, c3 = st.columns([1, 1, 1])
    use_stop = c1.checkbox("Quitar stopwords (es)", False, key="sim_stop")
    use_emb = c2.checkbox("Incluir embeddings (descarga modelo)", False)
    emb_name = c3.selectbox("Modelo de embeddings", EMBEDDING_MODELS, key="sim_emb", disabled=not use_emb)
    stop = STOP_ES if use_stop else None
    metrics = [m for m in SIM_METRICS if use_emb or m != "Coseno (embeddings)"]

    labels = [f"D{i + 1}" for i in range(len(docs))]
    mats: dict[str, tuple[np.ndarray, str]] = {}
    for mname in metrics:
        try:
            mats[mname] = compute_matrix(mname, docs, stop, emb_name)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{mname}: {type(e).__name__}: {e}")

    st.subheader("Comparar dos documentos")
    a, b = st.columns(2)
    i = a.selectbox("Documento A", range(len(docs)), format_func=lambda k: f"D{k + 1}: {docs[k][:60]}", index=0)
    j = b.selectbox("Documento B", range(len(docs)), format_func=lambda k: f"D{k + 1}: {docs[k][:60]}", index=1)
    st.dataframe(pd.DataFrame([{"métrica": k, "valor": round(float(M[i, j]), 4), "tipo": t}
                               for k, (M, t) in mats.items()]), hide_index=True)

    st.subheader("Matriz completa")
    if mats:
        pick = st.selectbox("Métrica", list(mats))
        M, kind = mats[pick]
        fig = px.imshow(M, x=labels, y=labels, text_auto=".2f", aspect="auto",
                        color_continuous_scale="Blues" if kind == "similitud" else "Reds",
                        title=f"{pick} ({kind})")
        st.plotly_chart(fig)
    with st.expander("Documentos"):
        for l, d in zip(labels, docs):
            st.write(f"**{l}**: {d}")


def tab_emb(docs: list[str]) -> None:
    if len(docs) < 2:
        st.warning("Necesitas al menos 2 documentos en el corpus.")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    choice = c1.selectbox("Modelo de embeddings (local)", EMBEDDING_MODELS + ["Otro..."])
    name = c1.text_input("ID del modelo (Hugging Face)", placeholder="p. ej. BAAI/bge-small-en-v1.5") if choice == "Otro..." else choice
    normalize = c2.checkbox("Normalizar (L2)", True)
    proj = c3.radio("Proyección 2D", ["PCA", "t-SNE"], horizontal=True)

    if st.button("🧬 Calcular embeddings", type="primary", disabled=not name.strip()):
        try:
            with st.spinner("Calculando embeddings..."):
                E = embed_cached(tuple(docs), name.strip(), normalize)
            st.session_state["emb"] = {"docs": tuple(docs), "name": name.strip(), "normalize": normalize, "E": E}
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudieron calcular los embeddings: {type(e).__name__}: {e}")
    emb = st.session_state.get("emb")
    if not emb or emb["docs"] != tuple(docs):
        st.info("Pulsa «Calcular embeddings» (la primera vez se descarga el modelo).")
        return
    E, labels = emb["E"], [f"D{i + 1}" for i in range(len(docs))]
    m = st.columns(3)
    m[0].metric("Documentos", E.shape[0])
    m[1].metric("Dimensión", E.shape[1])
    m[2].metric("Norma media", f"{np.linalg.norm(E, axis=1).mean():.3f}")

    k = st.slider("Dimensiones a mostrar", 4, min(64, E.shape[1]), min(8, E.shape[1]))
    st.dataframe(pd.DataFrame(E[:, :k], index=labels).round(4))

    S = cosine_similarity(E)
    st.plotly_chart(px.imshow(S, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Blues",
                              title="Similitud coseno entre embeddings"))

    use_proj = proj
    if proj == "t-SNE" and len(docs) < 4:
        st.info("t-SNE necesita ≥ 4 documentos; se usa PCA.")
        use_proj = "PCA"
    P = (PCA(n_components=2).fit_transform(E) if use_proj == "PCA" else
         TSNE(n_components=2, perplexity=max(1, min(5, len(docs) - 1)), init="pca", random_state=0).fit_transform(E))
    dfp = pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "doc": labels, "texto": [d[:80] for d in docs]})
    st.plotly_chart(px.scatter(dfp, x="x", y="y", text="doc", hover_data=["texto"], title=f"Proyección 2D ({use_proj})"))

    st.subheader("Búsqueda semántica")
    q = st.text_input("Consulta", "un animal doméstico descansando")
    if q.strip():
        try:
            qe = embed_cached((q,), emb["name"], emb["normalize"])
            sc = cosine_similarity(qe, E)[0]
            order = np.argsort(-sc)
            st.dataframe(pd.DataFrame({"doc": [labels[i] for i in order], "similitud": sc[order].round(4),
                                       "texto": [docs[i] for i in order]}), hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))


def tab_gen(cfg: SimpleNamespace) -> None:
    options = chat_options(cfg)
    if not options:
        st.info("Conecta tu API key para generar texto.")
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        system = st.text_area("System prompt", "Eres un asistente útil y conciso. Responde en español.", height=80)
        prompt = st.text_area("Prompt", "Explica en tres frases qué es un token en un LLM.", height=140)
    with c2:
        model = st.selectbox("Modelo", options)
        temperature = st.slider("temperature", 0.0, 2.0, 0.7, 0.05)
        top_p = st.slider("top_p", 0.0, 1.0, 1.0, 0.05)
        max_tokens = int(st.number_input("max_completion_tokens", 16, 32768, 2048, 64,
                                         help="En modelos de razonamiento incluye los tokens de razonamiento."))
    base: dict = {"temperature": temperature, "top_p": top_p, "max_completion_tokens": max_tokens}

    with st.expander("Parámetros avanzados"):
        a1, a2, a3 = st.columns(3)
        fp = a1.slider("frequency_penalty", -2.0, 2.0, 0.0, 0.1)
        pp = a1.slider("presence_penalty", -2.0, 2.0, 0.0, 0.1)
        seed_on = a2.checkbox("Fijar seed", False)
        seed = a2.number_input("seed", 0, 2**31 - 1, 42, disabled=not seed_on)
        stop_txt = a3.text_input("stop (separadas por coma, máx. 4)")
        if "gpt-oss" in model:
            effort = a3.select_slider("reasoning_effort", ["low", "medium", "high"], value="medium", key=f"eff_{model}")
            incl = a3.checkbox("Incluir razonamiento en la respuesta", True, key=f"inc_{model}")
            base.update(reasoning_effort=effort, include_reasoning=incl)
        elif "qwen3" in model:
            base["reasoning_effort"] = a3.selectbox("reasoning_effort", ["default", "none"], key=f"effq_{model}")
    if fp:
        base["frequency_penalty"] = fp
    if pp:
        base["presence_penalty"] = pp
    if seed_on:
        base["seed"] = int(seed)
    stops = [s.strip() for s in stop_txt.split(",") if s.strip()][:4]
    if stops:
        base["stop"] = stops
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    if st.button("✨ Generar", type="primary", disabled=not prompt.strip()):
        try:
            with st.spinner("Generando..."):
                st.session_state["gen"] = groq_generate(cfg.api_key, model, msgs, base) | {"model": model}
        except Exception as e:  # noqa: BLE001
            st.error(f"Error al llamar a Groq: {e}")
    gen = st.session_state.get("gen")
    if gen:
        show_result(gen)

    st.divider()
    st.subheader("Comparar modelos y temperaturas")
    cm = st.multiselect("Modelos", options, default=options[:1], key="cmp_models")
    temps_txt = st.text_input("Temperaturas (separadas por coma)", "0.0, 0.7, 1.3")
    if st.button("⚖️ Comparar"):
        try:
            temps = [float(t) for t in temps_txt.split(",") if t.strip()]
        except ValueError:
            st.error("Temperaturas inválidas.")
            return
        combos = [(m, t) for m in cm for t in temps][:9]
        results = []
        bar = st.progress(0.0)
        for n, (m, t) in enumerate(combos):
            try:
                p = {k: v for k, v in base.items() if k not in ("reasoning_effort", "include_reasoning")}
                if "gpt-oss" in m:
                    p.update({k: base[k] for k in ("reasoning_effort", "include_reasoning") if k in base})
                p["temperature"] = t
                results.append((m, t, groq_generate(cfg.api_key, m, msgs, p), None))
            except Exception as e:  # noqa: BLE001
                results.append((m, t, None, str(e)))
            bar.progress((n + 1) / len(combos))
        st.session_state["cmp"] = results
    for row_start in range(0, len(st.session_state.get("cmp", [])), 3):
        cols = st.columns(3)
        for col, (m, t, res, err) in zip(cols, st.session_state["cmp"][row_start:row_start + 3]):
            with col.container(border=True):
                st.markdown(f"**{m}** · T={t}")
                if err:
                    st.error(err)
                else:
                    st.write(res["text"] or "_(respuesta vacía)_")
                    st.caption(f"{res['completion_tokens']} tokens · {res['seconds']:.1f} s")


def show_result(gen: dict) -> None:
    if gen["dropped"]:
        st.warning("El modelo rechazó estos parámetros y se omitieron: " + ", ".join(gen["dropped"]))
    if not gen["text"].strip():
        st.warning("La respuesta llegó vacía" + (": se agotó max_completion_tokens (el razonamiento consume parte). "
                                                  "Súbelo o baja reasoning_effort." if gen["finish"] == "length" else "."))
    st.markdown(gen["text"])
    if gen.get("reasoning"):
        with st.expander("Razonamiento del modelo"):
            st.write(gen["reasoning"])
    c = st.columns(5)
    c[0].metric("Tokens entrada", gen["prompt_tokens"])
    c[1].metric("Tokens salida", gen["completion_tokens"])
    c[2].metric("Tiempo", f"{gen['seconds']:.2f} s")
    c[3].metric("Tokens/s", "—" if not gen["tps"] else f"{gen['tps']:.0f}")
    c[4].metric("finish_reason", gen["finish"])


def main() -> None:
    st.set_page_config(page_title="Laboratorio LLM con Groq", page_icon="🧪", layout="wide")
    st.title("🧪 Laboratorio de LLMs con Groq")
    cfg = sidebar()
    with st.expander("📚 Corpus de trabajo (un documento por línea) · lo usan Bag of Words, Similitud y Embeddings"):
        corpus = st.text_area("Documentos", DEFAULT_CORPUS, height=150, label_visibility="collapsed")
    docs = parse_corpus(corpus)
    st.caption(f"{len(docs)} documentos en el corpus")

    t_mod, t_tok, t_bow, t_sim, t_emb, t_gen = st.tabs(
        ["🧠 Modelos", "🔤 Tokens e IDs", "🎒 Bag of Words", "📏 Similitud", "🧬 Embeddings", "✍️ Generación"])
    with t_mod:
        tab_models(cfg)
    with t_tok:
        tab_tokens()
    with t_bow:
        tab_bow(docs)
    with t_sim:
        tab_sim(docs)
    with t_emb:
        tab_emb(docs)
    with t_gen:
        tab_gen(cfg)


if __name__ == "__main__":
    main()
