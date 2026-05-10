import ast
import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.getLogger("neo4j").setLevel(logging.ERROR)


_ENV_LOADED = False


def _load_dotenv_if_exists(base_dir: str) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    env_path = Path(base_dir).resolve() / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k:
                os.environ[k] = v
    except Exception:
        pass


class EmbeddingEncoder:
    """Text embedding encoder based on HuggingFace models (BGE/M3E compatible)."""

    def __init__(self, model_name: str, max_length: int = 256, local_files_only: bool = True) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_length = max(32, min(int(max_length), 1024))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=bool(local_files_only),
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=bool(local_files_only),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 24) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        torch = self._torch
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = [str(t or "") for t in texts[start : start + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                hidden = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            arr = pooled.detach().cpu().numpy().astype(np.float32)
            arr = self._l2_normalize(arr)
            vectors.append(arr)

        return np.vstack(vectors) if vectors else np.zeros((0, 384), dtype=np.float32)

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / denom


class FallbackHashEncoder:
    """Fallback encoder when HF model cannot be loaded."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = int(dim)
        self.model_name = "fallback-hash-encoder"

    def encode(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        del batch_size
        mats = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            chars = list(str(text or ""))
            if not chars:
                continue
            for ch in chars:
                idx = (ord(ch) * 1315423911) % self.dim
                mats[i, idx] += 1.0
            denom = np.linalg.norm(mats[i]) + 1e-12
            mats[i] /= denom
        return mats


class SQLiteVectorStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn_lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    source TEXT,
                    meta_json TEXT,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_meta (
                    meta_key TEXT PRIMARY KEY,
                    meta_value TEXT
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_vector_chunks_source ON vector_chunks(source)")
            conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        with self._conn() as conn:
            row = conn.execute("SELECT meta_value FROM vector_meta WHERE meta_key = ?", (str(key),)).fetchone()
        return str(row["meta_value"]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO vector_meta(meta_key, meta_value) VALUES (?, ?)
                ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value
                """,
                (str(key), str(value)),
            )
            conn.commit()

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT count(*) AS c FROM vector_chunks").fetchone()
        if not row:
            return 0
        return int(row["c"] or 0)

    def rebuild(self, chunks: List[Dict[str, Any]], vectors: np.ndarray) -> None:
        if len(chunks) != int(vectors.shape[0]):
            raise ValueError("chunks/vectors length mismatch")
        with self._conn_lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM vector_chunks")
                rows = []
                for i, chunk in enumerate(chunks):
                    vec = np.asarray(vectors[i], dtype=np.float32)
                    rows.append(
                        (
                            str(chunk["chunk_id"]),
                            str(chunk.get("text", "") or ""),
                            str(chunk.get("source", "") or ""),
                            json.dumps(chunk.get("meta", {}), ensure_ascii=False),
                            vec.tobytes(),
                            int(vec.shape[0]),
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO vector_chunks(chunk_id, text_content, source, meta_json, embedding, dim)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

    def load_matrix(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT chunk_id, text_content, source, meta_json, embedding, dim FROM vector_chunks"
            ).fetchall()

        metas: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        for row in rows:
            dim = int(row["dim"])
            vec = np.frombuffer(row["embedding"], dtype=np.float32, count=dim)
            vectors.append(vec)
            try:
                meta_obj = json.loads(row["meta_json"] or "{}")
            except Exception:
                meta_obj = {}
            metas.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "text": str(row["text_content"] or ""),
                    "source": str(row["source"] or ""),
                    "meta": meta_obj,
                }
            )
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32), []
        return np.vstack(vectors).astype(np.float32), metas


class HybridVectorRetriever:
    """Embedding + SQLite vector store + cosine Top-K retrieval."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        _load_dotenv_if_exists(base_dir)
        self.data_dir = os.path.join(base_dir, "data")
        self.db_path = os.path.join(self.data_dir, "vector_chunks.db")
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip() or "BAAI/bge-small-zh-v1.5"
        self.local_files_only = str(os.getenv("EMBEDDING_LOCAL_ONLY", "1")).strip().lower() not in {"0", "false", "no"}
        self.use_hf = str(os.getenv("EMBEDDING_USE_HF", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "24") or 24)
        self.top_k_default = int(os.getenv("VECTOR_TOP_K", "5") or 5)
        self.max_corpus_docs = int(os.getenv("VECTOR_MAX_DOCS", "1200") or 1200)

        self.store = SQLiteVectorStore(self.db_path)
        self._encoder: Optional[Any] = None
        self._index_lock = threading.Lock()
        self._matrix_cache: Optional[np.ndarray] = None
        self._meta_cache: Optional[List[Dict[str, Any]]] = None
        self._last_error: Optional[str] = None

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        if not self.use_hf:
            self._last_error = "EMBEDDING_USE_HF=0，使用 fallback 编码器。"
            self._encoder = FallbackHashEncoder()
            return self._encoder
        try:
            self._encoder = EmbeddingEncoder(model_name=self.model_name, local_files_only=self.local_files_only)
            self._last_error = None
            return self._encoder
        except Exception as exc:
            self._last_error = f"Embedding model load failed, fallback encoder enabled: {exc}"
            self._encoder = FallbackHashEncoder()
            return self._encoder

    def _source_signature(self) -> str:
        candidates = [
            os.path.join(self.data_dir, "medical_corpus.jsonl"),
            os.path.join(self.data_dir, "medical_new_2.json"),
            os.path.join(self.data_dir, "medical.json"),
        ]
        parts = [f"model={self.model_name}"]
        for p in candidates:
            if os.path.exists(p):
                st = os.stat(p)
                parts.append(f"{p}:{int(st.st_mtime)}:{int(st.st_size)}")
        raw = "|".join(parts)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _build_chunks(self) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunks.extend(self._chunks_from_neo4j(max_docs=self.max_corpus_docs))
        jsonl_path = os.path.join(self.data_dir, "medical_corpus.jsonl")
        if not chunks and os.path.exists(jsonl_path):
            chunks.extend(self._chunks_from_jsonl(jsonl_path, max_docs=self.max_corpus_docs))
        if not chunks:
            chunks.extend(self._chunks_from_medical_json(max_docs=self.max_corpus_docs))
        return chunks[: max(200, self.max_corpus_docs)]

    def _chunks_from_neo4j(self, max_docs: int) -> List[Dict[str, Any]]:
        from neo4j import GraphDatabase

        uri = (
            os.getenv("NEO4J_URI")
            or os.getenv("NEO4J_URL")
            or os.getenv("NEO4J_WEBSITE")
            or "bolt://127.0.0.1:7687"
        ).strip()
        user = (os.getenv("NEO4J_USER") or "neo4j").strip()
        password = (os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or "").strip()
        database = (os.getenv("NEO4J_DATABASE") or os.getenv("NEO4J_DBNAME") or "neo4j").strip() or "neo4j"
        if not password:
            return []

        candidates = [uri]
        if uri.startswith("neo4j://"):
            host = uri[len("neo4j://") :].split("/", 1)[0].split(":", 1)[0]
            candidates.insert(0, f"bolt://{host}:7687")
            candidates = [candidates[0]]
        if "bolt://127.0.0.1:7687" not in candidates:
            candidates.append("bolt://127.0.0.1:7687")

        rows = None
        for candidate in candidates:
            try:
                driver = GraphDatabase.driver(candidate, auth=(user, password), connection_timeout=5)
                with driver.session(database=database) as session:
                    rows = session.run(
                        """
                        MATCH (n)
                        WITH n,
                             toString(coalesce(n.name, n['\u540d\u79f0'], '')) AS nname,
                             CASE WHEN size(labels(n)) > 0 THEN labels(n)[0] ELSE 'Entity' END AS ntype
                        WHERE nname <> ''
                        LIMIT $limit
                        OPTIONAL MATCH (n)-[r]->(m)
                        WITH nname, ntype,
                             collect(
                                DISTINCT (
                                    type(r) + '->' + toString(coalesce(m.name, m['\u540d\u79f0'], elementId(m)))
                                )
                             )[0..8] AS rels
                        RETURN nname AS name, ntype AS ntype, rels AS rels
                        """,
                        limit=max_docs,
                    ).data()
                driver.close()
                break
            except Exception:
                rows = None
                continue

        if not rows:
            return []

        out: List[Dict[str, Any]] = []
        for i, row in enumerate(rows, start=1):
            name = str(row.get("name") or "").strip()
            ntype = str(row.get("ntype") or "Entity").strip()
            rels = row.get("rels") or []
            if not isinstance(rels, list):
                rels = []
            rel_text = "；".join([str(x).strip() for x in rels if str(x).strip()][:8])
            text = f"实体类型：{ntype}\n实体名称：{name}"
            if rel_text:
                text += f"\n相关关系：{rel_text}"
            chunk_id = hashlib.md5(f"neo4j:{i}:{name}:{ntype}".encode("utf-8")).hexdigest()
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": "neo4j",
                    "meta": {"name": name, "type": ntype, "row": i},
                }
            )
        return out

    def _chunks_from_jsonl(self, path: str, max_docs: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    for idx, raw in enumerate(f, start=1):
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        title = str(obj.get("title", "") or "").strip()
                        disease = str(obj.get("disease", "") or "").strip()
                        category = str(obj.get("category", "") or "").strip()
                        source = str(obj.get("source", "") or "").strip()
                        body = str(obj.get("text", "") or "").strip()
                        merged = "；".join([x for x in [title, disease, category, source] if x])
                        full_text = f"{merged}\n{body}".strip()
                        if not full_text:
                            continue
                        for ci, part in enumerate(self._split_text(full_text), start=1):
                            chunk_id = hashlib.md5(f"{path}:{idx}:{ci}".encode("utf-8")).hexdigest()
                            out.append(
                                {
                                    "chunk_id": chunk_id,
                                    "text": part,
                                    "source": source or "medical_corpus.jsonl",
                                    "meta": {"title": title, "disease": disease, "category": category, "row": idx},
                                }
                            )
                            if len(out) >= max_docs:
                                return out
                if out:
                    return out
            except Exception:
                continue
        return out

    def _chunks_from_medical_json(self, max_docs: int) -> List[Dict[str, Any]]:
        candidates = [
            os.path.join(self.data_dir, "medical_new_2.json"),
            os.path.join(self.data_dir, "medical.json"),
        ]
        records: List[Dict[str, Any]] = []
        for path in candidates:
            if not os.path.exists(path):
                continue
            for enc in ("utf-8", "utf-8-sig", "gbk"):
                try:
                    with open(path, "r", encoding=enc) as f:
                        for raw in f:
                            line = str(raw or "").strip().rstrip(",")
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                try:
                                    obj = ast.literal_eval(line)
                                except Exception:
                                    continue
                            if isinstance(obj, dict):
                                records.append(obj)
                    if records:
                        break
                except Exception:
                    continue
            if records:
                break

        out: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records, start=1):
            disease = str(rec.get("name", "") or "").strip()
            if not disease:
                continue
            text = self._medical_record_to_text(rec)
            for ci, part in enumerate(self._split_text(text), start=1):
                chunk_id = hashlib.md5(f"medical:{idx}:{ci}:{disease}".encode("utf-8")).hexdigest()
                out.append(
                    {
                        "chunk_id": chunk_id,
                        "text": part,
                        "source": "medical_new_2.json",
                        "meta": {"disease": disease, "row": idx},
                    }
                )
                if len(out) >= max_docs:
                    return out
        return out

    @staticmethod
    def _medical_record_to_text(rec: Dict[str, Any]) -> str:
        disease = str(rec.get("name", "") or "").strip()
        desc = str(rec.get("desc", "") or "").strip()
        cause = str(rec.get("cause", "") or "").strip()
        prevent = str(rec.get("prevent", "") or "").strip()
        cure_last = str(rec.get("cure_lasttime", "") or "").strip()
        cured_prob = str(rec.get("cured_prob", "") or "").strip()
        easy_get = str(rec.get("easy_get", "") or "").strip()

        def join_list(key: str, max_items: int = 12) -> str:
            value = rec.get(key, [])
            if not isinstance(value, list):
                return ""
            return "、".join([str(x).strip() for x in value if str(x).strip()][:max_items])

        parts = [
            f"疾病：{disease}" if disease else "",
            f"简介：{desc}" if desc else "",
            f"病因：{cause}" if cause else "",
            f"预防：{prevent}" if prevent else "",
            f"治疗周期：{cure_last}" if cure_last else "",
            f"治愈概率：{cured_prob}" if cured_prob else "",
            f"易感人群：{easy_get}" if easy_get else "",
            f"常用药：{join_list('common_drug')}" if join_list("common_drug") else "",
            f"推荐药：{join_list('recommand_drug')}" if join_list("recommand_drug") else "",
            f"症状：{join_list('symptom')}" if join_list("symptom") else "",
            f"检查：{join_list('check')}" if join_list("check") else "",
            f"治疗方法：{join_list('cure_way')}" if join_list("cure_way") else "",
        ]
        return "\n".join([p for p in parts if p]).strip()

    @staticmethod
    def _split_text(text: str, chunk_size: int = 260, overlap: int = 40) -> List[str]:
        content = str(text or "").strip()
        if not content:
            return []
        if len(content) <= chunk_size:
            return [content]
        out: List[str] = []
        step = max(20, chunk_size - overlap)
        for i in range(0, len(content), step):
            part = content[i : i + chunk_size].strip()
            if len(part) >= 12:
                out.append(part)
            if i + chunk_size >= len(content):
                break
        return out

    def ensure_index(self, force: bool = False) -> None:
        with self._index_lock:
            signature = self._source_signature()
            stored_signature = self.store.get_meta("source_signature", "")
            stored_model = self.store.get_meta("embedding_model", "")
            need_rebuild = force or (self.store.count() == 0) or (signature != stored_signature) or (
                stored_model != self.model_name
            )
            if not need_rebuild:
                if self._matrix_cache is None or self._meta_cache is None:
                    self._matrix_cache, self._meta_cache = self.store.load_matrix()
                return

            chunks = self._build_chunks()
            if not chunks:
                self._matrix_cache = np.zeros((0, 1), dtype=np.float32)
                self._meta_cache = []
                self.store.set_meta("source_signature", signature)
                self.store.set_meta("embedding_model", self.model_name)
                return

            encoder = self._get_encoder()
            vectors = encoder.encode([c["text"] for c in chunks], batch_size=self.batch_size)
            self.store.rebuild(chunks, vectors)
            self.store.set_meta("source_signature", signature)
            self.store.set_meta("embedding_model", getattr(encoder, "model_name", self.model_name))
            self._matrix_cache, self._meta_cache = vectors, chunks

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        text = str(query or "").strip()
        if not text:
            return []
        self.ensure_index(force=False)
        if self._matrix_cache is None or self._meta_cache is None or self._matrix_cache.shape[0] == 0:
            return []

        encoder = self._get_encoder()
        q_vec = encoder.encode([text], batch_size=1)[0]
        mat = self._matrix_cache
        sims = np.dot(mat, q_vec)

        k = max(1, min(int(top_k or self.top_k_default), len(sims)))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        sorted_idx = top_idx[np.argsort(-sims[top_idx])]

        hits: List[Dict[str, Any]] = []
        for idx in sorted_idx:
            score = float(sims[idx])
            meta = self._meta_cache[int(idx)]
            hits.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                    "source": meta.get("source", ""),
                    "meta": meta.get("meta", {}),
                    "score": round(score, 6),
                }
            )
        return hits

    def status(self) -> Dict[str, Any]:
        backend = "hf" if self.use_hf else "fallback"
        return {
            "enabled": True,
            "backend": backend,
            "embedding_model": self.model_name,
            "chunks": self.store.count(),
            "last_error": self._last_error,
        }
