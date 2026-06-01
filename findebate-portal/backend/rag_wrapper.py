import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


BACKEND_DIR = Path(__file__).resolve().parent
CHROMA_PATH = os.getenv(
    "FINDEBATE_CHROMA_PATH",
    str(BACKEND_DIR / "findebate_chromadb"),
)
RETRIEVAL_MODE = os.getenv("FINDEBATE_RETRIEVAL_MODE", "sqlite").strip().lower()

SOURCE_ALIASES = {
    "ABM": ["abm", "abm industries"],
    "AME": ["ame", "ametek"],
    "CFR": ["cfr", "cullen frost", "cullen/frost", "frost bank"],
    "CMI": ["cmi", "cummins"],
    "CPF": ["cpf", "central pacific"],
    "DE": ["de", "deere", "john deere"],
    "DNB": ["dnb", "dun bradstreet", "dun & bradstreet"],
    "DOV": ["dov", "dover"],
    "DX": ["dx", "dynex"],
    "ETN": ["etn", "eaton"],
    "FAF": ["faf", "first american"],
    "FIS": ["fis", "fidelity national information services", "fiserv"],
    "FN": ["fn", "fabrinet"],
    "FSS": ["fss", "federal signal"],
    "GCO": ["gco", "genesco"],
    "GD": ["gd", "general dynamics"],
    "GLW": ["glw", "corning"],
    "GNW": ["gnw", "genworth"],
    "HR": ["hr", "healthcare realty"],
    "HTH": ["hth", "hilltop"],
    "JBL": ["jbl", "jabil"],
    "KMT": ["kmt", "kennametal"],
    "KW": ["kw", "kennedy wilson"],
    "LH": ["lh", "labcorp", "laboratory corporation"],
    "LNN": ["lnn", "lindsay"],
    "LYB": ["lyb", "lyondellbasell"],
    "MDT": ["mdt", "medtronic"],
    "MKC": ["mkc", "mccormick"],
    "MSI": ["msi", "motorola solutions"],
    "MYE": ["mye", "myers"],
    "NEE": ["nee", "nextera"],
    "NPO": ["npo", "enpro"],
    "OHI": ["ohi", "omega healthcare"],
    "PCAR": ["pcar", "paccar"],
    "RPM": ["rpm", "rpm international"],
    "SF": ["sf", "stifel"],
    "SWN": ["swn", "southwestern energy"],
    "SYY": ["syy", "sysco"],
    "TK": ["tk", "teekay"],
    "TT": ["tt", "trane", "trane technologies"],
    "UNH": ["unh", "unitedhealth", "united health"],
    "UVE": ["uve", "universal insurance"],
    "VMI": ["vmi", "valmont"],
    "VSH": ["vsh", "vishay"],
    "WWW": ["www", "wolverine"],
    "WYNN": ["wynn", "wynn resorts"],
}
DIMENSION_QUERIES = {
    "general_financial": [
        "financial performance revenue earnings beat miss surprise results",
        "guidance outlook forecast expectations future performance strategy",
    ],
    "specialized_metrics": [
        "net interest margin NIM loan deposits credit quality asset quality",
        "return on assets ROA return on equity ROE efficiency ratio capital adequacy",
    ],
    "market_sentiment_risk": [
        "management confidence sentiment optimistic cautious positive negative tone",
        "risks challenges concerns headwinds uncertainties market conditions",
    ],
    "multi_query_integration": [
        "short-term immediate near-term weekly monthly quarterly timeline",
        "comprehensive integrated multi-dimensional longitudinal tracking",
    ],
}

AGENT_MAP = {
    "earnings_agent": ["general_financial", "specialized_metrics"],
    "market_agent": ["general_financial", "multi_query_integration"],
    "sentiment_agent": ["market_sentiment_risk"],
    "valuation_agent": ["specialized_metrics", "general_financial"],
    "risk_agent": ["market_sentiment_risk", "specialized_metrics"],
}

_collection = None
_model = None
_model_error = None


def get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection("findebate_rag")
    return _collection


def get_model():
    global _model, _model_error
    if _model is None and _model_error is None:
        from sentence_transformers import SentenceTransformer

        try:
            _model = SentenceTransformer("FinLang/finance-embeddings-investopedia")
        except Exception as exc:
            _model_error = str(exc)
            raise
    if _model_error:
        raise RuntimeError(_model_error)
    return _model


def _tokenize(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", query) if len(token) > 2]


def _db_path() -> Path:
    return Path(CHROMA_PATH) / "chroma.sqlite3"


def _source_files() -> list[str]:
    db_path = _db_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source_file'"
        ).fetchall()
    finally:
        conn.close()
    return sorted(row[0] for row in rows if row and row[0])


def infer_source_file(query: str, ticker: str | None = None) -> str:
    sources = _source_files()
    if not sources:
        return ""

    ticker = (ticker or "").strip().upper()
    if ticker:
        for source in sources:
            if source.upper().startswith(f"{ticker}_"):
                return source

    normalized = " ".join(_tokenize(query))
    normalized_tokens = set(normalized.split())
    best_source, best_score = "", 0
    for source in sources:
        source_upper = source.upper()
        source_ticker = source_upper.split("_", 1)[0]
        score = 0
        if source.lower() in query.lower():
            score += 20
        if source_ticker.lower() in normalized_tokens or re.search(rf"\b{re.escape(source_ticker)}\b", query, re.I):
            score += 12
        for alias in SOURCE_ALIASES.get(source_ticker, []):
            alias_tokens = _tokenize(alias)
            if alias and re.search(rf"\b{re.escape(alias.lower())}\b", query.lower()):
                score += 10 + len(alias_tokens)
            elif alias_tokens and all(token in normalized_tokens for token in alias_tokens):
                score += 6 + len(alias_tokens)
        if score > best_score:
            best_source, best_score = source, score
    return best_source if best_score else ""


def _retrieve_sqlite(query: str, top_k: int = 5, doc_type_filter=None, source_file_filter=None) -> list:
    db_path = Path(CHROMA_PATH) / "chroma.sqlite3"
    tokens = _tokenize(query)
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    rows = cursor.execute(
        """
        SELECT
            doc.id,
            doc.string_value AS chunk,
            source.string_value AS source_file,
            chunk_index.int_value AS chunk_id,
            type_meta.string_value AS type
        FROM embedding_metadata doc
        JOIN embedding_metadata source
            ON source.id = doc.id AND source.key = 'source_file'
        LEFT JOIN embedding_metadata chunk_index
            ON chunk_index.id = doc.id AND chunk_index.key = 'chunk_index'
        LEFT JOIN embedding_metadata type_meta
            ON type_meta.id = doc.id AND type_meta.key = 'type'
        WHERE doc.key = 'chroma:document'
        """
    ).fetchall()
    conn.close()

    scored = []
    fallback = []
    for _, chunk, source_file, chunk_id, chunk_type in rows:
        if doc_type_filter and chunk_type != doc_type_filter:
            continue
        if source_file_filter and source_file != source_file_filter:
            continue
        item = {
            "chunk": chunk or "",
            "source_file": source_file or "unknown",
            "chunk_id": chunk_id or 0,
            "type": chunk_type or "unknown",
            "score": 0.25 if source_file_filter else 0.1,
            "retrieval": "sqlite_source_fallback" if source_file_filter else "sqlite_keyword_fallback",
        }
        fallback.append(item)
        text = (chunk or "").lower()
        score = sum(text.count(token) for token in tokens)
        if source_file and source_file.lower().split("_")[0] in tokens:
            score += 3
        if source_file_filter:
            score += 1
        if score <= 0:
            continue
        scored.append(
            {
                "chunk": chunk or "",
                "source_file": source_file or "unknown",
                "chunk_id": chunk_id or 0,
                "type": chunk_type or "unknown",
                "score": round(min(0.99, 0.35 + score / 20), 4),
                "retrieval": "sqlite_keyword",
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    if scored:
        return scored[:top_k]
    return sorted(fallback, key=lambda item: item["chunk_id"])[:top_k]


def retrieve(query: str, top_k: int = 5, doc_type_filter=None, source_file_filter=None) -> list:
    if RETRIEVAL_MODE != "vector":
        return _retrieve_sqlite(
            query,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            source_file_filter=source_file_filter,
        )

    try:
        collection = get_collection()
        model = get_model()
        q_emb = model.encode([query], convert_to_numpy=True).tolist()
        where = {"type": doc_type_filter} if doc_type_filter else None
        results = collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if source_file_filter and meta.get("source_file") != source_file_filter:
                continue
            output.append(
                {
                    "chunk": doc,
                    "source_file": meta["source_file"],
                    "chunk_id": meta.get("chunk_index", meta.get("chunk_id", "")),
                    "type": meta.get("type", "unknown"),
                    "score": round(1 - dist, 4),
                    "retrieval": "chroma_vector",
                }
            )
        return output
    except Exception:
        return _retrieve_sqlite(
            query,
            top_k=top_k,
            doc_type_filter=doc_type_filter,
            source_file_filter=source_file_filter,
        )


def get_agent_context(source_file=None, top_k=3) -> dict:
    context = {}
    for agent, dims in AGENT_MAP.items():
        seen, chunks = set(), []
        for dim in dims:
            for query in DIMENSION_QUERIES[dim]:
                for result in retrieve(query, top_k=top_k * 5, source_file_filter=source_file):
                    uid = f"{result['source_file']}_chunk_{result['chunk_id']}"
                    if uid not in seen:
                        seen.add(uid)
                        chunks.append(result)
        chunks.sort(key=lambda item: item["score"], reverse=True)
        context[agent] = chunks[:top_k]
    return context


def chunks_to_text(chunks: list) -> str:
    return "\n\n".join(f"[Chunk {idx + 1}]: {chunk['chunk']}" for idx, chunk in enumerate(chunks))


def merge_chunks(primary: list, fallback: list, limit: int = 5) -> list:
    seen = set()
    merged = []
    for chunk in [*(primary or []), *(fallback or [])]:
        uid = f"{chunk.get('source_file')}_chunk_{chunk.get('chunk_id')}"
        if uid in seen:
            continue
        seen.add(uid)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged
