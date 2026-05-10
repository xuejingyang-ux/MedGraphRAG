import ast
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from neo4j import GraphDatabase

try:
    import jieba  # type: ignore
except Exception:
    jieba = None

logging.getLogger("neo4j").setLevel(logging.ERROR)


_ENV_LOADED = False


def _load_dotenv_if_exists(force: bool = False) -> None:
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return
    _ENV_LOADED = True

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value
    except Exception:
        pass


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_list(value: Any) -> List[str]:
    items: List[str] = []
    if value is None:
        return items
    if isinstance(value, list):
        source = value
    else:
        source = [value]
    for item in source:
        if isinstance(item, list):
            for sub in item:
                text = str(sub or "").strip()
                if text:
                    items.append(text)
            continue
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _unique_preserve_order(items: Iterable[str], max_items: int = 0) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if max_items > 0 and len(out) >= max_items:
            break
    return out


class KnowledgeGraphManager:
    def __init__(self) -> None:
        _load_dotenv_if_exists()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.database = os.getenv("NEO4J_DATABASE", os.getenv("NEO4J_DBNAME", "neo4j")).strip() or "neo4j"
        try:
            self.connection_timeout = float(os.getenv("NEO4J_CONNECTION_TIMEOUT", "5") or 5)
        except Exception:
            self.connection_timeout = 5.0
        try:
            self.reconnect_cooldown = float(os.getenv("NEO4J_RECONNECT_COOLDOWN", "60") or 60)
        except Exception:
            self.reconnect_cooldown = 60.0
        self.driver = None
        self._connect_error: Optional[str] = None
        self._last_connect_failed_at = 0.0

        self._fallback_graph = self._build_local_graph()
        self._connect_neo4j()

    def _connect_neo4j(self) -> None:
        raw_uri = (
            os.getenv("NEO4J_URI")
            or os.getenv("NEO4J_URL")
            or os.getenv("NEO4J_WEBSITE")
            or ""
        ).strip()
        user = (os.getenv("NEO4J_USER") or "neo4j").strip()
        raw_password = (os.getenv("NEO4J_PASSWORD") or os.getenv("NEO4J_PASS") or "").strip()

        uri_candidates = self._build_uri_candidates(raw_uri)
        password_candidates = [raw_password] if raw_password else []
        # Backward-compatible default from this project; can be overridden by env.
        if "wei8kang7.long" not in password_candidates:
            password_candidates.append("wei8kang7.long")

        errors: List[str] = []
        for uri in uri_candidates:
            for password in password_candidates:
                if not password:
                    continue
                try:
                    driver = GraphDatabase.driver(
                        uri,
                        auth=(user, password),
                        connection_timeout=self.connection_timeout,
                    )
                    driver.verify_connectivity()
                    self.driver = driver
                    self._connect_error = None
                    self._last_connect_failed_at = 0.0
                    return
                except Exception as exc:
                    errors.append(f"{uri} ({user}): {exc}")
                    continue

        self.driver = None
        if not raw_password:
            self._connect_error = (
                "Neo4j connection failed. NEO4J_PASSWORD not set; tried project default password but still failed. "
                f"Tried URIs: {', '.join(uri_candidates)}"
            )
        else:
            sample_error = errors[0] if errors else "unknown error"
            self._connect_error = f"Neo4j connection failed. First error: {sample_error}"
        self._last_connect_failed_at = time.time()

    def _build_uri_candidates(self, raw_uri: str) -> List[str]:
        candidates: List[str] = []

        def add(value: str) -> None:
            uri = str(value or "").strip()
            if not uri or uri in candidates:
                return
            candidates.append(uri)

        if raw_uri:
            if raw_uri.startswith("http://"):
                host = raw_uri[len("http://"):].split("/", 1)[0].split(":", 1)[0]
                add(f"bolt://{host}:7687")
            elif raw_uri.startswith("https://"):
                host = raw_uri[len("https://"):].split("/", 1)[0].split(":", 1)[0]
                add(f"bolt+s://{host}:7687")
            elif raw_uri.startswith("neo4j://"):
                host = raw_uri[len("neo4j://"):].split("/", 1)[0].split(":", 1)[0]
                add(f"bolt://{host}:7687")
            else:
                add(raw_uri)

        add("bolt://127.0.0.1:7687")
        add("bolt://localhost:7687")
        return candidates

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass

    def _session(self):
        if self.driver is None:
            raise RuntimeError("Neo4j is not connected.")
        return self.driver.session(database=self.database)

    def check_connection(self) -> bool:
        if self.driver is None:
            if self._connect_error and self._last_connect_failed_at:
                if (time.time() - self._last_connect_failed_at) < self.reconnect_cooldown:
                    return False
            _load_dotenv_if_exists(force=True)
            self._connect_neo4j()
            if self.driver is None:
                return False
        try:
            with self._session() as session:
                session.run("RETURN 1 AS ok").single()
            return True
        except Exception as exc:
            self._connect_error = str(exc)
            self._last_connect_failed_at = time.time()
            try:
                self.close()
            except Exception:
                pass
            self.driver = None
            _load_dotenv_if_exists(force=True)
            self._connect_neo4j()
            if self.driver is None:
                return False
            try:
                with self._session() as session:
                    session.run("RETURN 1 AS ok").single()
                self._connect_error = None
                return True
            except Exception as exc2:
                self._connect_error = str(exc2)
                return False

    def get_kg_statistics(self) -> Dict[str, int]:
        if self.check_connection():
            try:
                with self._session() as session:
                    rec = session.run(
                        """
                        CALL () {
                            MATCH (n)
                            RETURN count(n) AS entities
                        }
                        CALL () {
                            MATCH ()-[r]->()
                            RETURN count(r) AS relationships
                        }
                        RETURN entities, relationships
                        """
                    ).single()
                if rec:
                    return {
                        "entities": int(rec.get("entities", 0) or 0),
                        "relationships": int(rec.get("relationships", 0) or 0),
                    }
            except Exception as exc:
                self._connect_error = str(exc)

        return {
            "entities": len(self._fallback_graph.get("nodes", [])),
            "relationships": len(self._fallback_graph.get("links", [])),
        }

    def query_whole_graph(self, limit: int = 300) -> Dict[str, List[Dict[str, Any]]]:
        limit = max(20, min(int(limit or 300), 1500))
        if self.check_connection():
            try:
                with self._session() as session:
                    rows = session.run(
                        """
                        MATCH (a)-[r]->(b)
                        RETURN
                            elementId(a) AS source_id,
                            toString(coalesce(properties(a)['name'], properties(a)['title'], properties(a)['\u540d\u79f0'], elementId(a))) AS source_name,
                            CASE WHEN size(labels(a)) > 0 THEN labels(a)[0] ELSE 'Entity' END AS source_type,
                            elementId(b) AS target_id,
                            toString(coalesce(properties(b)['name'], properties(b)['title'], properties(b)['\u540d\u79f0'], elementId(b))) AS target_name,
                            CASE WHEN size(labels(b)) > 0 THEN labels(b)[0] ELSE 'Entity' END AS target_type,
                            type(r) AS rel_type
                        LIMIT $limit
                        """,
                        limit=limit,
                    ).data()
                return self._rows_to_graph(rows)
            except Exception as exc:
                self._connect_error = str(exc)

        return self._slice_fallback_graph(limit=limit)

    def search_nodes(self, keyword: str) -> Dict[str, List[Dict[str, Any]]]:
        query = str(keyword or "").strip()
        if not query:
            return {"nodes": [], "links": []}

        linked_entities = self.link_query_entities(text=query, max_entities=3)
        if linked_entities:
            seed_ids = []
            for item in linked_entities[:1]:
                seed_id = str(item.get("id") or "").strip()
                if seed_id and seed_id not in seed_ids:
                    seed_ids.append(seed_id)
            if seed_ids:
                try:
                    graph = self.query_multi_hop_subgraph(seed_ids=seed_ids, hops=1, max_edges=140)
                    if graph.get("nodes"):
                        graph["focus_ids"] = seed_ids[:3]
                        return graph
                except Exception as exc:
                    self._connect_error = str(exc)

        if self.check_connection():
            try:
                with self._session() as session:
                    rows = session.run(
                        """
                        MATCH (n)
                        WITH n, properties(n) AS p
                        WHERE toLower(toString(coalesce(p['name'], p['title'], p['\u540d\u79f0'], ''))) CONTAINS toLower($q)
                        WITH n
                        ORDER BY
                            CASE WHEN toLower(toString(coalesce(properties(n)['name'], properties(n)['title'], properties(n)['\u540d\u79f0'], ''))) = toLower($q) THEN 0 ELSE 1 END,
                            size(labels(n)) DESC
                        LIMIT 3
                        OPTIONAL MATCH (n)-[r]-(m)
                        RETURN
                            elementId(n) AS n_id,
                            toString(coalesce(properties(n)['name'], properties(n)['title'], properties(n)['\u540d\u79f0'], elementId(n))) AS n_name,
                            CASE WHEN size(labels(n)) > 0 THEN labels(n)[0] ELSE 'Entity' END AS n_type,
                            CASE WHEN m IS NULL THEN '' ELSE elementId(m) END AS m_id,
                            CASE WHEN m IS NULL THEN '' ELSE toString(coalesce(properties(m)['name'], properties(m)['title'], properties(m)['\u540d\u79f0'], elementId(m))) END AS m_name,
                            CASE WHEN m IS NULL OR size(labels(m)) = 0 THEN 'Entity' ELSE labels(m)[0] END AS m_type,
                            CASE WHEN r IS NULL THEN '' ELSE type(r) END AS rel_type,
                            CASE WHEN r IS NULL THEN true ELSE elementId(startNode(r)) = elementId(n) END AS n_to_m
                        LIMIT 120
                        """,
                        q=query,
                    ).data()

                graph = self._search_rows_to_graph(rows)
                focus_ids = []
                for row in rows[:1]:
                    nid = str(row.get("n_id") or "").strip()
                    if nid and nid not in focus_ids:
                        focus_ids.append(nid)
                graph["focus_ids"] = focus_ids
                return graph
            except Exception as exc:
                self._connect_error = str(exc)

        return self._search_fallback_graph(query=query)

    def link_query_entities(self, text: str, max_entities: int = 4) -> List[Dict[str, Any]]:
        keywords = self._extract_keywords(text, limit=12)
        if not keywords:
            return []
        max_entities = max(1, min(int(max_entities or 4), 12))
        lowered = [k.lower() for k in keywords]

        if self.check_connection():
            try:
                with self._session() as session:
                    rows = session.run(
                        """
                        MATCH (n)
                        WITH n, properties(n) AS p, toLower(toString(coalesce(p['name'], p['title'], p['\u540d\u79f0'], ''))) AS nname
                        WHERE any(k IN $keywords WHERE nname CONTAINS k)
                        WITH n, nname,
                             reduce(score = 0, k IN $keywords |
                                 score + CASE
                                     WHEN nname = k THEN 10
                                     WHEN nname STARTS WITH k THEN 6
                                     WHEN nname CONTAINS k THEN 3
                                     ELSE 0
                                 END
                             ) AS score
                        RETURN
                            elementId(n) AS entity_id,
                            toString(coalesce(properties(n)['name'], properties(n)['title'], properties(n)['\u540d\u79f0'], elementId(n))) AS entity_name,
                            CASE WHEN size(labels(n)) > 0 THEN labels(n)[0] ELSE 'Entity' END AS entity_type,
                            score AS score
                        ORDER BY score DESC, size(nname) ASC
                        LIMIT $limit
                        """,
                        keywords=lowered,
                        limit=max_entities,
                    ).data()
                return [
                    {
                        "id": str(r.get("entity_id") or "").strip(),
                        "name": str(r.get("entity_name") or "").strip(),
                        "type": str(r.get("entity_type") or "Entity"),
                        "score": float(r.get("score") or 0.0),
                    }
                    for r in rows
                    if str(r.get("entity_id") or "").strip()
                ]
            except Exception as exc:
                self._connect_error = str(exc)

        # Fallback entity link based on local graph name matching.
        linked: List[Dict[str, Any]] = []
        for node in self._fallback_graph.get("nodes", []):
            name = str(node.get("name") or "").strip()
            if not name:
                continue
            nlow = name.lower()
            score = 0
            for kw in lowered:
                if nlow == kw:
                    score += 10
                elif nlow.startswith(kw):
                    score += 6
                elif kw in nlow:
                    score += 3
            if score <= 0:
                continue
            linked.append(
                {
                    "id": str(node.get("id") or "").strip(),
                    "name": name,
                    "type": str(node.get("type") or "Entity"),
                    "score": float(score),
                }
            )
        linked.sort(key=lambda x: (-x["score"], len(x["name"])))
        return linked[:max_entities]

    def query_multi_hop_subgraph(self, seed_ids: List[str], hops: int = 2, max_edges: int = 260) -> Dict[str, List[Dict[str, Any]]]:
        seeds = [str(x or "").strip() for x in (seed_ids or []) if str(x or "").strip()]
        if not seeds:
            return {"nodes": [], "links": []}
        hops = max(1, min(int(hops or 2), 3))
        max_edges = max(40, min(int(max_edges or 260), 1500))

        if self.check_connection():
            try:
                return self._expand_multi_hop_from_seed_neo4j(seeds=seeds, hops=hops, max_edges=max_edges)
            except Exception as exc:
                self._connect_error = str(exc)

        return self._expand_multi_hop_from_seed_fallback(seeds=seeds, hops=hops, max_edges=max_edges)

    def _expand_multi_hop_from_seed_neo4j(self, seeds: List[str], hops: int, max_edges: int) -> Dict[str, List[Dict[str, Any]]]:
        node_map: Dict[str, Dict[str, Any]] = {}
        edge_seen: Set[Tuple[str, str, str]] = set()
        links: List[Dict[str, Any]] = []

        frontier = list(seeds)
        visited: Set[str] = set(seeds)

        for _ in range(hops):
            if not frontier or len(links) >= max_edges:
                break

            with self._session() as session:
                rows = session.run(
                    """
                    UNWIND $seed_ids AS sid
                    MATCH (s)
                    WHERE elementId(s) = sid
                    MATCH (s)-[r]-(t)
                    RETURN
                        elementId(s) AS sid,
                        toString(coalesce(properties(s)['name'], properties(s)['title'], properties(s)['\u540d\u79f0'], elementId(s))) AS s_name,
                        CASE WHEN size(labels(s)) > 0 THEN labels(s)[0] ELSE 'Entity' END AS s_type,
                        elementId(t) AS tid,
                        toString(coalesce(properties(t)['name'], properties(t)['title'], properties(t)['\u540d\u79f0'], elementId(t))) AS t_name,
                        CASE WHEN size(labels(t)) > 0 THEN labels(t)[0] ELSE 'Entity' END AS t_type,
                        type(r) AS rel_type,
                        elementId(startNode(r)) AS rel_start_id,
                        elementId(endNode(r)) AS rel_end_id
                    LIMIT $limit
                    """,
                    seed_ids=frontier,
                    limit=max_edges * 2,
                ).data()

            next_frontier: List[str] = []
            for row in rows:
                sid = str(row.get("sid") or "").strip()
                tid = str(row.get("tid") or "").strip()
                if not sid or not tid:
                    continue
                s_name = str(row.get("s_name") or sid).strip()
                t_name = str(row.get("t_name") or tid).strip()
                s_type = str(row.get("s_type") or "Entity").strip()
                t_type = str(row.get("t_type") or "Entity").strip()
                rel_type = str(row.get("rel_type") or "RELATED").strip() or "RELATED"
                rel_start = str(row.get("rel_start_id") or sid).strip()
                rel_end = str(row.get("rel_end_id") or tid).strip()

                node_map[sid] = {"id": sid, "name": s_name, "type": s_type}
                node_map[tid] = {"id": tid, "name": t_name, "type": t_type}

                edge_key = (rel_start, rel_end, rel_type)
                if edge_key not in edge_seen:
                    edge_seen.add(edge_key)
                    links.append({"source_id": rel_start, "target_id": rel_end, "type": rel_type})

                if tid not in visited:
                    visited.add(tid)
                    next_frontier.append(tid)
                if sid not in visited:
                    visited.add(sid)
                    next_frontier.append(sid)
                if len(links) >= max_edges:
                    break

            frontier = _unique_preserve_order(next_frontier, max_items=200)

        return {"nodes": list(node_map.values()), "links": links[:max_edges]}

    def _expand_multi_hop_from_seed_fallback(self, seeds: List[str], hops: int, max_edges: int) -> Dict[str, List[Dict[str, Any]]]:
        node_map = {str(n.get("id")): n for n in self._fallback_graph.get("nodes", [])}
        links = self._fallback_graph.get("links", [])
        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        for link in links:
            sid = str(link.get("source_id") or "")
            tid = str(link.get("target_id") or "")
            if not sid or not tid:
                continue
            adjacency.setdefault(sid, []).append(link)
            adjacency.setdefault(tid, []).append(link)

        visited: Set[str] = set(seeds)
        frontier = list(seeds)
        out_nodes: Dict[str, Dict[str, Any]] = {}
        out_links: List[Dict[str, Any]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        for _ in range(hops):
            if not frontier or len(out_links) >= max_edges:
                break
            next_frontier: List[str] = []
            for nid in frontier:
                for link in adjacency.get(nid, []):
                    sid = str(link.get("source_id") or "")
                    tid = str(link.get("target_id") or "")
                    rel = str(link.get("type") or "RELATED")
                    if not sid or not tid:
                        continue
                    key = (sid, tid, rel)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        out_links.append({"source_id": sid, "target_id": tid, "type": rel})
                    if sid in node_map:
                        out_nodes[sid] = node_map[sid]
                    if tid in node_map:
                        out_nodes[tid] = node_map[tid]
                    other = tid if sid == nid else sid
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append(other)
                    if len(out_links) >= max_edges:
                        break
                if len(out_links) >= max_edges:
                    break
            frontier = _unique_preserve_order(next_frontier, max_items=200)

        return {"nodes": list(out_nodes.values()), "links": out_links[:max_edges]}

    @staticmethod
    def _rank_records_by_relevance(
        records: List[Dict[str, Any]],
        linked_entities: List[Dict[str, Any]],
        keywords: List[str],
    ) -> List[Dict[str, Any]]:
        if not records:
            return []
        linked_names = {str(e.get("name") or "").strip().lower() for e in linked_entities}
        linked_names.discard("")
        kws = [str(k).strip().lower() for k in keywords if str(k).strip()]

        def score(rec: Dict[str, Any]) -> float:
            src = str(rec.get("source") or "").strip().lower()
            tgt = str(rec.get("target") or "").strip().lower()
            rel = str(rec.get("relationship") or "").strip().lower()
            val = 0.0
            if src in linked_names:
                val += 6.0
            if tgt in linked_names:
                val += 6.0
            for kw in kws:
                if kw and kw in src:
                    val += 2.0
                if kw and kw in tgt:
                    val += 2.0
                if kw and kw in rel:
                    val += 1.2
            if "symptom" in rel or "症状" in rel:
                val += 0.8
            if "drug" in rel or "药" in rel:
                val += 0.8
            if "check" in rel or "检查" in rel:
                val += 0.6
            return val

        return sorted(records, key=score, reverse=True)

    def process_user_query(
        self,
        text: str,
        save_to_db: bool = False,
        depth: int = 2,
        similarity_threshold: float = 0.68,
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        del save_to_db, similarity_threshold
        keywords = self._extract_keywords(text, limit=max(6, top_k * 2))
        if not keywords:
            return []

        hops = max(1, min(int(depth or 2), 3))
        limit = max(40, min(top_k * 30, 420))

        # 1) Entity Linking + Multi-hop neighborhood retrieval.
        linked_entities = self.link_query_entities(text=text, max_entities=max(2, min(top_k, 6)))
        if linked_entities:
            seed_ids = [str(item.get("id") or "").strip() for item in linked_entities if str(item.get("id") or "").strip()]
            if seed_ids:
                multi_hop_graph = self.query_multi_hop_subgraph(seed_ids=seed_ids, hops=hops, max_edges=max(140, top_k * 45))
                records = self._records_from_graph(multi_hop_graph)
                records = self._rank_records_by_relevance(records, linked_entities=linked_entities, keywords=keywords)
                records = self._dedupe_records(records)
                if records:
                    return records[: max(24, top_k * 12)]

        # 2) Fallback: keyword edge retrieval.
        if self.check_connection():
            try:
                with self._session() as session:
                    rows = session.run(
                        """
                        MATCH (a)-[r]->(b)
                        WITH a, b, r, properties(a) AS ap, properties(b) AS bp
                        WHERE any(k IN $keywords WHERE
                            toLower(toString(coalesce(ap['name'], ap['title'], ap['\u540d\u79f0'], ''))) CONTAINS k OR
                            toLower(toString(coalesce(bp['name'], bp['title'], bp['\u540d\u79f0'], ''))) CONTAINS k OR
                            toLower(type(r)) CONTAINS k
                        )
                        RETURN
                            toString(coalesce(ap['name'], ap['title'], ap['\u540d\u79f0'], elementId(a))) AS source,
                            CASE WHEN size(labels(a)) > 0 THEN labels(a)[0] ELSE 'Entity' END AS source_type,
                            type(r) AS relationship,
                            toString(coalesce(bp['name'], bp['title'], bp['\u540d\u79f0'], elementId(b))) AS target,
                            CASE WHEN size(labels(b)) > 0 THEN labels(b)[0] ELSE 'Entity' END AS target_type
                        LIMIT $limit
                        """,
                        keywords=[k.lower() for k in keywords],
                        limit=limit,
                    ).data()
                deduped = self._dedupe_records(rows)
                deduped = self._rank_records_by_relevance(deduped, linked_entities=linked_entities, keywords=keywords)
                return deduped[: max(20, top_k * 8)]
            except Exception as exc:
                self._connect_error = str(exc)

        # 3) Local fallback.
        merged_records: List[Dict[str, Any]] = []
        for kw in keywords:
            graph = self._search_fallback_graph(query=kw)
            merged_records.extend(self._records_from_graph(graph))
            if len(merged_records) >= limit:
                break
        deduped = self._dedupe_records(merged_records)
        deduped = self._rank_records_by_relevance(deduped, linked_entities=linked_entities, keywords=keywords)
        return deduped[: max(20, top_k * 8)]

    def _extract_keywords(self, text: str, limit: int = 8) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        stop_words = {
            "怎么",
            "如何",
            "哪些",
            "什么",
            "可以",
            "需要",
            "一下",
            "请问",
            "请",
            "是否",
            "就是",
            "这个",
            "那个",
            "以及",
            "还有",
            "会不会",
            "会",
            "吗",
            "呢",
            "啊",
            "呀",
            "和",
            "与",
            "及",
            "的",
        }

        tokens: List[str] = []
        if jieba is not None:
            try:
                tokens.extend([str(t).strip() for t in jieba.lcut(raw) if str(t).strip()])
            except Exception:
                pass
        tokens.extend(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", raw))

        cleaned: List[str] = []
        for tok in tokens:
            t = str(tok or "").strip()
            if len(t) < 2:
                continue
            if t in stop_words:
                continue
            # Remove question suffixes like "怎么治疗".
            t = re.sub(r"(怎么|如何|哪些|什么|可以|需要|吗|呢|呀|啊)+$", "", t).strip()
            if len(t) < 2 or t in stop_words:
                continue
            cleaned.append(t)
        return _unique_preserve_order(cleaned, max_items=limit)

    def _rows_to_graph(self, rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        node_map: Dict[str, Dict[str, Any]] = {}
        link_seen: Set[Tuple[str, str, str]] = set()
        links: List[Dict[str, Any]] = []

        for row in rows:
            sid = str(row.get("source_id") or "").strip()
            tid = str(row.get("target_id") or "").strip()
            if not sid or not tid:
                continue
            sname = str(row.get("source_name") or sid).strip()
            tname = str(row.get("target_name") or tid).strip()
            stype = str(row.get("source_type") or "Entity").strip()
            ttype = str(row.get("target_type") or "Entity").strip()
            rtype = str(row.get("rel_type") or "RELATED").strip() or "RELATED"

            node_map[sid] = {"id": sid, "name": sname, "type": stype}
            node_map[tid] = {"id": tid, "name": tname, "type": ttype}

            key = (sid, tid, rtype)
            if key in link_seen:
                continue
            link_seen.add(key)
            links.append({"source_id": sid, "target_id": tid, "type": rtype})

        return {"nodes": list(node_map.values()), "links": links}

    def _search_rows_to_graph(self, rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        node_map: Dict[str, Dict[str, Any]] = {}
        link_seen: Set[Tuple[str, str, str]] = set()
        links: List[Dict[str, Any]] = []

        for row in rows:
            n_id = str(row.get("n_id") or "").strip()
            if not n_id:
                continue
            node_map[n_id] = {
                "id": n_id,
                "name": str(row.get("n_name") or n_id).strip(),
                "type": str(row.get("n_type") or "Entity").strip(),
            }

            m_id = str(row.get("m_id") or "").strip()
            rel_type = str(row.get("rel_type") or "").strip()
            if not m_id or not rel_type:
                continue

            node_map[m_id] = {
                "id": m_id,
                "name": str(row.get("m_name") or m_id).strip(),
                "type": str(row.get("m_type") or "Entity").strip(),
            }

            n_to_m = bool(row.get("n_to_m", True))
            sid, tid = (n_id, m_id) if n_to_m else (m_id, n_id)
            key = (sid, tid, rel_type)
            if key in link_seen:
                continue
            link_seen.add(key)
            links.append({"source_id": sid, "target_id": tid, "type": rel_type})

        return {"nodes": list(node_map.values()), "links": links}

    def _records_from_graph(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = {str(n.get("id")): n for n in graph.get("nodes", [])}
        records: List[Dict[str, Any]] = []
        for link in graph.get("links", []):
            sid = str(link.get("source_id") or "")
            tid = str(link.get("target_id") or "")
            if not sid or not tid:
                continue
            source_node = nodes.get(sid, {})
            target_node = nodes.get(tid, {})
            records.append(
                {
                    "source": str(source_node.get("name") or sid),
                    "source_type": str(source_node.get("type") or "Entity"),
                    "relationship": str(link.get("type") or "RELATED"),
                    "target": str(target_node.get("name") or tid),
                    "target_type": str(target_node.get("type") or "Entity"),
                }
            )
        return records

    def _dedupe_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, str, str]] = set()
        out: List[Dict[str, Any]] = []
        for rec in records:
            source = str(rec.get("source") or "").strip()
            rel = str(rec.get("relationship") or "RELATED").strip()
            target = str(rec.get("target") or "").strip()
            if not source or not target:
                continue
            key = (source, rel, target)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "source": source,
                    "source_type": str(rec.get("source_type") or "Entity"),
                    "relationship": rel,
                    "target": target,
                    "target_type": str(rec.get("target_type") or "Entity"),
                }
            )
        return out

    def _slice_fallback_graph(self, limit: int = 300) -> Dict[str, List[Dict[str, Any]]]:
        links = self._fallback_graph.get("links", [])[:limit]
        keep_ids: Set[str] = set()
        for link in links:
            keep_ids.add(str(link.get("source_id")))
            keep_ids.add(str(link.get("target_id")))

        all_nodes = {str(n.get("id")): n for n in self._fallback_graph.get("nodes", [])}
        nodes = [all_nodes[nid] for nid in keep_ids if nid in all_nodes]
        if not nodes:
            nodes = self._fallback_graph.get("nodes", [])[: min(limit, 120)]
        return {"nodes": nodes, "links": links}

    def _search_fallback_graph(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        q = str(query or "").strip().lower()
        if not q:
            return {"nodes": [], "links": []}

        nodes = self._fallback_graph.get("nodes", [])
        links = self._fallback_graph.get("links", [])
        matched_ids = [
            str(node.get("id"))
            for node in nodes
            if q in str(node.get("name") or "").lower()
        ][:1]
        if not matched_ids:
            return {"nodes": [], "links": []}

        matched_set = set(matched_ids)
        selected_links: List[Dict[str, Any]] = []
        per_seed_counts: Dict[str, int] = {}
        for link in links:
            sid = str(link.get("source_id"))
            tid = str(link.get("target_id"))
            if sid not in matched_set and tid not in matched_set:
                continue
            seed = sid if sid in matched_set else tid
            per_seed_counts[seed] = per_seed_counts.get(seed, 0) + 1
            if per_seed_counts[seed] > 40:
                continue
            selected_links.append(link)
            if len(selected_links) >= 120:
                break

        node_ids = set(matched_set)
        for link in selected_links:
            node_ids.add(str(link.get("source_id")))
            node_ids.add(str(link.get("target_id")))

        node_map = {str(n.get("id")): n for n in nodes}
        selected_nodes = [node_map[nid] for nid in node_ids if nid in node_map]
        return {"nodes": selected_nodes, "links": selected_links, "focus_ids": matched_ids}

    def _build_local_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        source_candidates = [
            os.path.join(self.data_dir, "medical_new_2.json"),
            os.path.join(self.data_dir, "medical.json"),
        ]

        records: List[Dict[str, Any]] = []
        for path in source_candidates:
            if os.path.exists(path):
                records = self._load_medical_records(path)
                if records:
                    break

        node_map: Dict[str, Dict[str, Any]] = {}
        link_seen: Set[Tuple[str, str, str]] = set()
        links: List[Dict[str, Any]] = []

        def add_node(name: str, node_type: str) -> Optional[str]:
            clean_name = str(name or "").strip()
            if not clean_name:
                return None
            node_id = hashlib.md5(f"{node_type}:{clean_name}".encode("utf-8")).hexdigest()
            if node_id not in node_map:
                node_map[node_id] = {"id": node_id, "name": clean_name, "type": node_type}
            return node_id

        def add_edge(source_name: str, source_type: str, rel_type: str, target_name: str, target_type: str) -> None:
            sid = add_node(source_name, source_type)
            tid = add_node(target_name, target_type)
            if not sid or not tid:
                return
            key = (sid, tid, rel_type)
            if key in link_seen:
                return
            link_seen.add(key)
            links.append({"source_id": sid, "target_id": tid, "type": rel_type})

        for rec in records:
            disease = str(rec.get("name") or "").strip()
            if not disease:
                continue
            add_node(disease, "Disease")

            for symptom in _safe_list(rec.get("symptom")):
                add_edge(disease, "Disease", "has_symptom", symptom, "Symptom")

            for drug in _safe_list(rec.get("common_drug")):
                add_edge(disease, "Disease", "common_drug", drug, "Drug")
            for drug in _safe_list(rec.get("recommand_drug")):
                add_edge(disease, "Disease", "recommand_drug", drug, "Drug")

            for food in _safe_list(rec.get("do_eat")):
                add_edge(disease, "Disease", "do_eat", food, "Food")
            for food in _safe_list(rec.get("recommand_eat")):
                add_edge(disease, "Disease", "recommand_eat", food, "Food")
            for food in _safe_list(rec.get("not_eat")):
                add_edge(disease, "Disease", "no_eat", food, "Food")

            for check in _safe_list(rec.get("check")):
                add_edge(disease, "Disease", "need_check", check, "Check")

            for dept in _safe_list(rec.get("cure_department")):
                add_edge(disease, "Disease", "belongs_to", dept, "Department")

            for cure in _safe_list(rec.get("cure_way")):
                add_edge(disease, "Disease", "cure_way", cure, "Cure")

            companions = _safe_list(rec.get("acompany")) + _safe_list(rec.get("acompany_with"))
            for comorbidity in companions:
                add_edge(disease, "Disease", "acompany_with", comorbidity, "Disease")

            for detail in _safe_list(rec.get("drug_detail")):
                parts = [p.strip() for p in str(detail).split(",") if str(p).strip()]
                if len(parts) >= 2:
                    producer = parts[0]
                    drug = parts[1]
                    add_edge(producer, "Producer", "produces", drug, "Drug")

        return {"nodes": list(node_map.values()), "links": links}

    def _load_medical_records(self, path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(path, "r", encoding=enc) as f:
                    lines = f.readlines()
                for raw in lines:
                    line = raw.strip()
                    if not line:
                        continue
                    obj = self._parse_line_to_dict(line)
                    if obj:
                        records.append(obj)
                if records:
                    return records
            except Exception:
                continue
        return records

    def _parse_line_to_dict(self, line: str) -> Optional[Dict[str, Any]]:
        text = str(line or "").strip().rstrip(",")
        if not text:
            return None

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None
