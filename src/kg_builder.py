import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from knowledge_graph import KnowledgeGraphManager


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


class MedicalKnowledgeGraphBuilder:
    def __init__(self, jsonl_path: str, use_llm: bool = True) -> None:
        self.jsonl_path = str(jsonl_path or "").strip()
        env_use_llm = _to_bool(os.getenv("KG_USE_LLM_EXTRACTION", "1"), default=True)
        self.use_llm = bool(use_llm) and env_use_llm
        self.kg = KnowledgeGraphManager()
        self.driver = self.kg.driver
        self.database = self.kg.database

    def close(self) -> None:
        try:
            self.kg.close()
        except Exception:
            pass

    def build_graph(self, reset: bool = False, limit: Optional[int] = None) -> Dict[str, Any]:
        records = self._load_jsonl_records(self.jsonl_path, limit=limit)
        total_records = len(records)

        if self.driver is None:
            return {
                "total_records": total_records,
                "processed": 0,
                "failed": total_records,
                "documents_in_graph": 0,
                "entities_in_graph": 0,
                "mentions_in_graph": 0,
                "relations_in_graph": 0,
                "noise_nodes": 0,
                "verified": False,
                "sample_errors": ["Neo4j not connected. Please configure NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD."],
            }

        errors: List[str] = []
        processed = 0
        mentions_written = 0

        try:
            if reset:
                with self.driver.session(database=self.database) as session:
                    session.run(
                        """
                        MATCH (d:Document)
                        DETACH DELETE d
                        """
                    )

            with self.driver.session(database=self.database) as session:
                for rec in records:
                    try:
                        payload = self._build_document_payload(rec)
                        doc_id = payload["doc_id"]
                        title = payload["title"]
                        source = payload["source"]
                        text = payload["text"]
                        entities = self._extract_entities_from_record(rec, payload["full_text"])

                        session.run(
                            """
                            MERGE (d:Document {id: $doc_id})
                            SET d.title = $title,
                                d.source = $source,
                                d.text = $text,
                                d.updated_at = datetime()
                            """,
                            doc_id=doc_id,
                            title=title,
                            source=source,
                            text=text,
                        )

                        for name in entities:
                            session.run(
                                """
                                MATCH (d:Document {id: $doc_id})
                                MERGE (e:Entity {name: $name})
                                MERGE (d)-[:MENTIONS]->(e)
                                """,
                                doc_id=doc_id,
                                name=name,
                            )
                            mentions_written += 1

                        processed += 1
                    except Exception as exc:
                        errors.append(str(exc))

                stats = session.run(
                    """
                    CALL {
                        MATCH (d:Document)
                        RETURN count(d) AS documents_in_graph
                    }
                    CALL {
                        MATCH (e:Entity)
                        RETURN count(e) AS entities_in_graph
                    }
                    CALL {
                        MATCH (:Document)-[m:MENTIONS]->(:Entity)
                        RETURN count(m) AS mentions_in_graph
                    }
                    RETURN documents_in_graph, entities_in_graph, mentions_in_graph
                    """
                ).single()

                noise = session.run(
                    """
                    MATCH (n)
                    WHERE trim(toString(coalesce(n.name, n.title, ''))) =~ '^[0-9]+$'
                    RETURN count(n) AS c
                    """
                ).single()

            documents_in_graph = int((stats or {}).get("documents_in_graph", 0) or 0)
            entities_in_graph = int((stats or {}).get("entities_in_graph", 0) or 0)
            mentions_in_graph = int((stats or {}).get("mentions_in_graph", 0) or 0)
            noise_nodes = int((noise or {}).get("c", 0) or 0)

            return {
                "total_records": total_records,
                "processed": processed,
                "failed": max(0, total_records - processed),
                "documents_in_graph": documents_in_graph,
                "entities_in_graph": entities_in_graph,
                "mentions_in_graph": mentions_in_graph,
                "relations_in_graph": mentions_in_graph,
                "noise_nodes": noise_nodes,
                "verified": (processed == total_records and noise_nodes == 0),
                "sample_errors": errors[:10],
            }
        except Exception as exc:
            errors.append(str(exc))
            return {
                "total_records": total_records,
                "processed": processed,
                "failed": max(0, total_records - processed),
                "documents_in_graph": 0,
                "entities_in_graph": 0,
                "mentions_in_graph": mentions_written,
                "relations_in_graph": mentions_written,
                "noise_nodes": 0,
                "verified": False,
                "sample_errors": errors[:10],
            }

    def _load_jsonl_records(self, path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"JSONL path not found: {path}")

        max_records = int(limit or 0)
        records: List[Dict[str, Any]] = []
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            records.append(obj)
                            if max_records > 0 and len(records) >= max_records:
                                return records
                if records:
                    return records
            except Exception:
                records = []
                continue
        raise RuntimeError("Failed to read JSONL records. Please ensure file is valid UTF-8/GBK JSONL.")

    def _build_document_payload(self, rec: Dict[str, Any]) -> Dict[str, str]:
        title = str(rec.get("title", "") or "").strip()
        source = str(rec.get("source", "") or "").strip()
        body = str(rec.get("text", "") or "").strip()
        disease = str(rec.get("disease", "") or "").strip()
        category = str(rec.get("category", "") or "").strip()

        parts = []
        if title:
            parts.append(f"title: {title}")
        if disease:
            parts.append(f"disease: {disease}")
        if category:
            parts.append(f"category: {category}")
        if source:
            parts.append(f"source: {source}")
        if body:
            parts.append(body)
        full_text = "\n".join(parts).strip()

        if not full_text:
            full_text = json.dumps(rec, ensure_ascii=False)

        base = f"{title}|{source}|{full_text}"
        doc_id = hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]
        return {
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "text": body or full_text,
            "full_text": full_text,
        }

    def _extract_entities_from_record(self, rec: Dict[str, Any], text: str) -> List[str]:
        seed = []
        for key in ("disease", "title", "category"):
            value = str(rec.get(key, "") or "").strip()
            if value:
                seed.append(value)

        # KG_USE_LLM_EXTRACTION=0 is respected by self.use_llm; fallback extraction is rule based.
        chunks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text or "")
        entities = seed + chunks

        seen = set()
        out: List[str] = []
        for item in entities:
            name = str(item or "").strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
            if len(out) >= 20:
                break
        return out
