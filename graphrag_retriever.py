"""
GraphRAG-enhanced retrieval module (static knowledge base)

This module plugs into an existing FAISS-based RAG pipeline, adding a lightweight GraphRAG layer:
- Offline:
  1) chunk documents
  2) extract entity-relation triples (LLM-assisted recommended, rule-based fallback)
  3) build graph + discover communities
  4) generate community summaries (LLM-assisted)
  5) persist artifacts: graph_edges.jsonl, graph_nodes.json, communities.json, chunk_meta.jsonl

- Online:
  1) vector retrieve Top-K seed chunks
  2) expand by graph neighborhood + community membership
  3) optional hybrid scoring

For Chinese domain KBs (e.g., tea standards), RuleBasedExtractor is too weak.
This version adds a DeepSeek-chat based extractor + community summarizer.

Environment variables:
- DEEPSEEK_API_KEY              (required to enable LLM extractor/summarizer)
- GRAPHRAG_DEEPSEEK_BASE_URL    (default: https://api.deepseek.com)
- GRAPHRAG_DEEPSEEK_MODEL       (default: deepseek-chat)
- GRAPHRAG_USE_LLM_EXTRACTOR    (default: auto; set to "0" to force rule-based)
- GRAPHRAG_USE_LLM_SUMMARY      (default: auto; set to "0" to disable summaries)
- GRAPHRAG_LLM_MAX_TRIPLES      (default: 24)
- GRAPHRAG_LLM_MAX_ENTITY_LEN   (default: 16)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

# Optional: OpenAI-compatible client (DeepSeek provides OpenAI-style API)
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str = ""
    tags: Dict[str, str] = None


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    chunk_id: str


@dataclass
class Community:
    community_id: str
    nodes: List[str]
    chunk_ids: List[str]
    summary: str = ""


# -----------------------------
# Utilities
# -----------------------------

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", (s or "").strip()).strip()

def _first_json_obj_or_arr(s: str) -> Optional[str]:
    """
    Try to extract the first JSON object/array substring from a messy LLM output.
    """
    s = _strip_code_fences(s)
    if not s:
        return None
    # Prefer object
    start_candidates = []
    for ch in ("{", "["):
        idx = s.find(ch)
        if idx != -1:
            start_candidates.append((idx, ch))
    if not start_candidates:
        return None
    start_candidates.sort(key=lambda x: x[0])
    start, opener = start_candidates[0]
    closer = "}" if opener == "{" else "]"

    # Simple bracket matching
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None

def _safe_json_loads(s: str) -> Optional[Any]:
    s = _strip_code_fences(s)
    cand = _first_json_obj_or_arr(s) or s
    try:
        return json.loads(cand)
    except Exception:
        return None

def _norm_entity(e: str, max_len: int) -> str:
    e = (e or "").strip()
    e = re.sub(r"\s+", "", e)  # remove spaces/newlines
    # 清理转义换行
    e = e.replace("\\n", "").replace("\\r", "").replace("\\t", "")
    # trim punctuation ends (扩展了更多符号)
    e = e.strip(" ,;:，；：。()（）[]【】<>《》\"“”\'‘’\n\r\t!！#$%^&*+~`@")
    if len(e) > max_len:
        e = e[:max_len]
    return e

def _bad_entity(e: str) -> bool:
    if not e:
        return True
    if len(e) < 2:
        # 单字符实体（如 "!"、"#"、"("）一律过滤
        return True
    if "\n" in e or "\r" in e:
        return True
    # Reject if too long — looks like a paragraph
    if len(e) >= 24:
        return True
    # Long ASCII sequences are usually noise for Chinese domain entities
    if re.fullmatch(r"[A-Za-z0-9_./\-]{8,}", e or ""):
        return True
    # 纯标点 / 纯符号 / 纯数字 → 噪声
    if re.fullmatch(r"[\W\d_]+", e):
        return True
    # 含有 CJK 兼容字符区段（犌犅犜 等 OCR 乱码常出现在 U+7280-U+72FF 及类似段）
    # 检测：如果实体中"生僻字"（非常见 CJK 基本集高频字）占比 >50%，视为乱码
    if _is_garbled_chinese(e):
        return True
    # 以 \n 开头或结尾的片段
    if e.startswith("\\n") or e.endswith("\\n"):
        return True
    return False


# ---------------------
# OCR 乱码检测
# ---------------------
# PDF OCR 乱码特征：拉丁字母被错误映射到 CJK 码位（常见于 U+7280-U+72FF 犬部字符）
# 例如: "犌犅／犜" = "GB/T", "狊犲狀狊狅狉" = "sensor"
_GARBLED_CJK_RANGE = re.compile(r'[\u7280-\u72FF]')  # 犬/犭部：PDF乱码高发区
_GARBLED_CJK_RANGE2 = re.compile(r'[\u7240-\u727F]')  # 牛/牜部：另一个高发区

def _is_garbled_chinese(e: str) -> bool:
    """
    检测疑似 PDF-OCR 乱码的实体。
    策略：如果实体中超过 30% 的字符落在 CJK 乱码高发区（U+7280-U+72FF），
    则判定为 OCR 错误映射，而非有效实体。
    
    典型乱码: 犌犅犜 (=GB/T), 狊犲狀狊狅狉犲狏犪犾狌犪狋犻狅狀 (=sensoryevaluation)
    """
    if not e:
        return False
    total = len(e)
    if total < 2:
        return False
    garbled_count = len(_GARBLED_CJK_RANGE.findall(e)) + len(_GARBLED_CJK_RANGE2.findall(e))
    if garbled_count == 0:
        return False
    garbled_ratio = garbled_count / total
    # 如果 >30% 字符在乱码区域，认定为乱码
    return garbled_ratio > 0.3

def _norm_relation(r: str) -> str:
    r = (r or "").strip()
    r = re.sub(r"\s+", "", r)
    r = r.strip(" ,;:，；：。()（）[]【】<>《》\"“”'’\n\r\t")
    if not r:
        r = "相关"
    if len(r) > 8:
        r = r[:8]
    return r


# -----------------------------
# LLM-assisted extractor / summarizer (pluggable)
# -----------------------------

class EntityRelationExtractor:
    """
    Pluggable extractor.

    Output triples should be concise and consistent across the corpus:
      List[(head, relation, tail)]
    """
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        raise NotImplementedError


class RuleBasedExtractor(EntityRelationExtractor):
    """
    Very lightweight baseline (kept as fallback).
    """
    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        ents: List[str] = []
        for left, right in [("《", "》"), ("“", "”"), ("\"", "\"")]:
            start = 0
            while True:
                i = text.find(left, start)
                if i == -1:
                    break
                j = text.find(right, i + 1)
                if j == -1:
                    break
                ent = text[i + 1 : j].strip()
                if ent:
                    ents.append(ent)
                start = j + 1
        triples = []
        for i in range(len(ents) - 1):
            triples.append((ents[i], "关联", ents[i + 1]))
        return triples


class CommunitySummarizer:
    def summarize(self, community_id: str, nodes: List[str], chunk_texts: List[str]) -> str:
        raise NotImplementedError


class _DeepSeekClientFactory:
    @staticmethod
    def build_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
        if OpenAI is None:
            return None
        api_key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
        if not api_key:
            return None
        base_url = (base_url or os.getenv("GRAPHRAG_DEEPSEEK_BASE_URL", "https://api.deepseek.com")).strip()
        return OpenAI(api_key=api_key, base_url=base_url)


class DeepSeekChatExtractor(EntityRelationExtractor):
    """
    LLM-based entity+relation extractor for Chinese domain standards.

    It asks DeepSeek-chat to output structured triples and then applies strict post-processing
    to prevent "giant paragraph entities" and normalize relations.
    """
    def __init__(
        self,
        client: Any = None,
        model: Optional[str] = None,
        max_triples: Optional[int] = None,
        max_entity_len: Optional[int] = None,
        temperature: float = 0.0,
    ):
        self.client = client or _DeepSeekClientFactory.build_client()
        self.model = (model or os.getenv("GRAPHRAG_DEEPSEEK_MODEL", "deepseek-chat")).strip()
        self.max_triples = int(max_triples or os.getenv("GRAPHRAG_LLM_MAX_TRIPLES", "24"))
        self.max_entity_len = int(max_entity_len or os.getenv("GRAPHRAG_LLM_MAX_ENTITY_LEN", "16"))
        self.temperature = float(temperature)
        self._cache: Dict[str, List[Tuple[str, str, str]]] = {}

        # Relation vocabulary (LLM can choose others, but we encourage these)
        self.allowed_relations = [
            "属于", "包括", "定义", "用于", "审评方法", "指标", "要求", "计算", "权数为", "影响", "对比", "来源", "相关",
        ]

    def _enabled(self) -> bool:
        use = os.getenv("GRAPHRAG_USE_LLM_EXTRACTOR", "").strip()
        if use == "0":
            return False
        return self.client is not None

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        if not text or not text.strip():
            return []
        key = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]

        if not self._enabled():
            triples = RuleBasedExtractor().extract_triples(text)
            self._cache[key] = triples
            return triples

        sys_prompt = (
             "你是“茶标准知识图谱抽取器”。\n "
             "任务：从给定中文文本中抽取【领域实体】与【语义关系】三元组，用于构建知识图谱检索。\n\n "
             "领域实体示例：叶底、汤色、香气、滋味、绿茶、红茶、审评方法、感官审评、评分、权数、等级、样品、冲泡条件等。\n "
             "关系类型尽量从以下集合中选择（更有利于检索）： "
            +  "、 ".join(self.allowed_relations) +  "。\n\n "
             "硬性约束：\n "
             "1) 实体必须“短、准、可复用”，不要整段文本，不要带换行；长度≤{max_len}。\n "  # 保持单括号，这是变量
             "2) 不要抽取纯数字/纯符号/无意义片段。\n "
             "3) 只输出 JSON，不要输出解释。\n\n "
             "输出 JSON 结构（必须严格遵守）：\n "
             "{{\n "                                    # 修改为双括号
            '   "triples ": [\n'
            '    {{ "head ":  "... ",  "relation ":  "... ",  "tail ":  "... "}},\n' # 修改为双括号
             "    ...\n "
             "  ]\n "
             "}}\n "                                    # 修改为双括号
        ).format(max_len=self.max_entity_len)

        user_prompt = (
            "请对下面文本抽取三元组。最多输出 {max_triples} 条最有用、最领域相关的三元组。\n\n"
            "【文本】\n"
            "{text}\n"
        ).format(max_triples=self.max_triples, text=text.strip())

        raw = ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            # LLM failure -> fallback to rule-based for this chunk
            triples = RuleBasedExtractor().extract_triples(text)
            self._cache[key] = triples
            return triples

        data = _safe_json_loads(raw)
        triples_in: List[Dict[str, str]] = []
        if isinstance(data, dict) and isinstance(data.get("triples"), list):
            triples_in = [t for t in data["triples"] if isinstance(t, dict)]
        elif isinstance(data, list):
            # allow direct list format
            triples_in = [t for t in data if isinstance(t, dict)]
        else:
            triples_in = []

        triples_out: List[Tuple[str, str, str]] = []
        seen = set()
        for t in triples_in:
            h = _norm_entity(str(t.get("head", "")), self.max_entity_len)
            r = _norm_relation(str(t.get("relation", "")))
            tail = _norm_entity(str(t.get("tail", "")), self.max_entity_len)

            if _bad_entity(h) or _bad_entity(tail) or h == tail:
                continue
            # Soft-normalize relations (keep LLM output but clamp length)
            if r not in self.allowed_relations and len(r) <= 1:
                r = "相关"

            k = (h, r, tail)
            if k in seen:
                continue
            seen.add(k)
            triples_out.append(k)
            if len(triples_out) >= self.max_triples:
                break

        self._cache[key] = triples_out
        return triples_out


class DeepSeekCommunitySummarizer(CommunitySummarizer):
    """
    LLM-based community summarizer.

    Input: a community's node list + a few representative chunk snippets.
    Output: a short Chinese summary that can be used as "global context".
    """
    def __init__(
        self,
        client: Any = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_chars: int = 220,
    ):
        self.client = client or _DeepSeekClientFactory.build_client()
        self.model = (model or os.getenv("GRAPHRAG_DEEPSEEK_MODEL", "deepseek-chat")).strip()
        self.temperature = float(temperature)
        self.max_chars = int(max_chars)
        self._cache: Dict[str, str] = {}

    def _enabled(self) -> bool:
        use = os.getenv("GRAPHRAG_USE_LLM_SUMMARY", "").strip()
        if use == "0":
            return False
        return self.client is not None

    def summarize(self, community_id: str, nodes: List[str], chunk_texts: List[str]) -> str:
        if not self._enabled():
            return ""

        key = hashlib.md5(
            (community_id + "|" + ",".join(nodes[:60]) + "|" + str(len(chunk_texts))).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]

        nodes_short = "、".join(nodes[:30])
        snippets = []
        for t in chunk_texts[:4]:
            t = (t or "").strip()
            if not t:
                continue
            t = re.sub(r"\s+", " ", t)
            snippets.append(t[:420])
        snip_text = "\n".join([f"- {s}" for s in snippets])

        sys_prompt = (
            "你是“知识图谱社区摘要器”。\n"
            "给定一个社区的实体列表与若干文本片段，请生成一个可用于检索增强的摘要。\n\n"
            "要求：\n"
            "1) 用中文输出，{max_chars} 字以内。\n"
            "2) 摘要应概括该社区的主题/范围，并点出 3-8 个关键实体。\n"
            "3) 避免空泛套话，尽量写出标准条款/指标/方法的要点。\n"
            "4) 只输出摘要正文，不要标题，不要列表标号。\n"
        ).format(max_chars=self.max_chars)

        user_prompt = (
            f"社区ID: {community_id}\n"
            f"实体（部分）: {nodes_short}\n\n"
            f"相关片段:\n{snip_text}\n\n"
            "请生成摘要："
        )

        summary = ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            summary = (resp.choices[0].message.content or "").strip()
            summary = _strip_code_fences(summary)
            summary = re.sub(r"\s+", " ", summary).strip()
            if len(summary) > self.max_chars:
                summary = summary[: self.max_chars]
        except Exception:
            summary = ""

        self._cache[key] = summary
        return summary


# -----------------------------
# GraphRAG Indexer (offline)
# -----------------------------

class GraphRAGIndexer:
    """
    Offline index builder.

    Default behavior:
    - If DEEPSEEK_API_KEY exists and openai client is available, use DeepSeekChatExtractor
      and DeepSeekCommunitySummarizer automatically.
    - Otherwise fallback to RuleBasedExtractor and empty summaries.
    """
    def __init__(
        self,
        extractor: Optional[EntityRelationExtractor] = None,
        summarizer: Optional[CommunitySummarizer] = None,
        auto_llm: bool = True,
    ):
        if nx is None:
            raise ImportError("networkx is required for GraphRAGIndexer. Please `pip install networkx`.")

        self.chunk_meta: Dict[str, Dict[str, Any]] = {}
        self.chunk_text: Dict[str, str] = {}
        self.graph = nx.MultiDiGraph()
        self.triples: List[Triple] = []
        self.communities: List[Community] = []

        # Auto choose LLM extractor/summarizer if possible
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = RuleBasedExtractor()
            if auto_llm and os.getenv("GRAPHRAG_USE_LLM_EXTRACTOR", "").strip() != "0":
                ds = DeepSeekChatExtractor()
                if ds.client is not None:
                    self.extractor = ds

        if summarizer is not None:
            self.summarizer = summarizer
        else:
            self.summarizer = None
            if auto_llm and os.getenv("GRAPHRAG_USE_LLM_SUMMARY", "").strip() != "0":
                sm = DeepSeekCommunitySummarizer()
                if sm.client is not None:
                    self.summarizer = sm

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            self.chunk_meta[c.chunk_id] = {"source": c.source, "tags": c.tags or {}}
            self.chunk_text[c.chunk_id] = c.text or ""
            triples = self.extractor.extract_triples(c.text or "")
            for h, r, t in triples:
                self._add_triple(h, r, t, c.chunk_id)

    def _add_triple(self, head: str, relation: str, tail: str, chunk_id: str) -> None:
        head = (head or "").strip()
        tail = (tail or "").strip()
        relation = _norm_relation(relation)
        if not head or not tail:
            return
        if _bad_entity(head) or _bad_entity(tail) or head == tail:
            return

        self.graph.add_node(head)
        self.graph.add_node(tail)
        self.graph.add_edge(head, tail, relation=relation, chunk_id=chunk_id)
        self.triples.append(Triple(head=head, relation=relation, tail=tail, chunk_id=chunk_id))

    def build_communities(
        self,
        min_size: int = 5,
        summarize: bool = True,
        max_chunks_per_community: int = 4,
        sleep_s: float = 0.0,
    ) -> List[Community]:
        """
        Community discovery on the undirected projection (greedy modularity).

        If summarize=True and a summarizer is available, it will also populate Community.summary.
        """
        if self.graph.number_of_nodes() == 0:
            self.communities = []
            return self.communities

        undirected = nx.Graph()
        for u, v, _data in self.graph.edges(data=True):
            undirected.add_edge(u, v)

        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(undirected))

        communities: List[Community] = []
        for idx, nodes in enumerate(comms):
            nodes_list = sorted(list(nodes))
            if len(nodes_list) < min_size:
                continue

            chunk_ids: Set[str] = set()
            for u in nodes_list:
                for _, _, data in self.graph.out_edges(u, data=True):
                    chunk_ids.add(str(data.get("chunk_id", "")))
                for _, _, data in self.graph.in_edges(u, data=True):
                    chunk_ids.add(str(data.get("chunk_id", "")))

            cid_list = sorted([cid for cid in chunk_ids if cid])
            communities.append(
                Community(
                    community_id=f"c{idx}",
                    nodes=nodes_list,
                    chunk_ids=cid_list,
                    summary="",
                )
            )

        self.communities = communities

        if summarize and self.summarizer is not None and self.communities:
            for c in self.communities:
                sample_cids = c.chunk_ids[:max_chunks_per_community]
                texts = [self.chunk_text.get(cid, "") for cid in sample_cids if cid in self.chunk_text]
                c.summary = self.summarizer.summarize(c.community_id, c.nodes, texts) or ""
                if sleep_s > 0:
                    time.sleep(float(sleep_s))

        return self.communities

    def attach_community_summaries(self, summaries: Dict[str, str]) -> None:
        for c in self.communities:
            if c.community_id in summaries:
                c.summary = summaries[c.community_id]

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        # graph edges
        graph_path = os.path.join(out_dir, "graph_edges.jsonl")
        with open(graph_path, "w", encoding="utf-8") as f:
            for tr in self.triples:
                f.write(json.dumps(asdict(tr), ensure_ascii=False) + "\n")

        # node list
        nodes_path = os.path.join(out_dir, "graph_nodes.json")
        with open(nodes_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self.graph.nodes())), f, ensure_ascii=False, indent=2)

        # communities
        comm_path = os.path.join(out_dir, "communities.json")
        with open(comm_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.communities], f, ensure_ascii=False, indent=2)

        # chunk meta
        meta_path = os.path.join(out_dir, "chunk_meta.jsonl")
        with open(meta_path, "w", encoding="utf-8") as f:
            for cid, meta in self.chunk_meta.items():
                row = {"chunk_id": cid, **meta}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# GraphRAG Retriever (online)
# -----------------------------

class GraphRAGRetriever:
    """
    Online hybrid retriever.

    Inputs:
      - vector_hits: List[(chunk_id, score)] from FAISS (higher is better)
      - chunk_text_map: Dict[chunk_id -> text]
      - graph artifacts: out_dir from GraphRAGIndexer.save()
    """
    def __init__(self, artifact_dir: str):
        if nx is None:
            raise ImportError("networkx is required for GraphRAGRetriever. Please `pip install networkx`.")
        self.artifact_dir = artifact_dir
        self.graph = nx.MultiDiGraph()
        self.communities: Dict[str, Community] = {}
        self.node2communities: Dict[str, Set[str]] = {}
        self.chunk2nodes: Dict[str, Set[str]] = {}
        self._load()

    def _load(self) -> None:
        edges_path = os.path.join(self.artifact_dir, "graph_edges.jsonl")
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                tr = json.loads(line)
                h, r, t, cid = tr["head"], tr["relation"], tr["tail"], tr["chunk_id"]
                self.graph.add_node(h)
                self.graph.add_node(t)
                self.graph.add_edge(h, t, relation=r, chunk_id=cid)
                self.chunk2nodes.setdefault(cid, set()).update([h, t])

        comm_path = os.path.join(self.artifact_dir, "communities.json")
        if os.path.exists(comm_path):
            comms = json.load(open(comm_path, "r", encoding="utf-8"))
            for c in comms:
                com = Community(**c)
                self.communities[com.community_id] = com
                for n in com.nodes:
                    self.node2communities.setdefault(n, set()).add(com.community_id)

    def expand(
        self,
        vector_hits: List[Tuple[str, float]],
        chunk_text_map: Dict[str, str],
        top_seed: int = 5,
        hop: int = 1,
        max_expand: int = 12,
        w_vec: float = 0.7,
        w_graph: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Return hybrid context:
          - seeds: top vector chunks
          - expanded: graph-augmented chunks
          - community_summaries: matched community summaries
        """
        seeds = vector_hits[:top_seed]
        seed_chunk_ids = [cid for cid, _ in seeds]

        # Collect seed nodes
        seed_nodes: Set[str] = set()
        for cid in seed_chunk_ids:
            seed_nodes |= self.chunk2nodes.get(cid, set())

        # Neighborhood expansion
        cand_chunks: Set[str] = set(seed_chunk_ids)
        frontier = set(seed_nodes)
        visited_nodes = set(seed_nodes)
        for _ in range(max(1, hop)):
            next_frontier = set()
            for n in frontier:
                for _, v, data in self.graph.out_edges(n, data=True):
                    next_frontier.add(v)
                    cand_chunks.add(str(data.get("chunk_id", "")))
                for u, _, data in self.graph.in_edges(n, data=True):
                    next_frontier.add(u)
                    cand_chunks.add(str(data.get("chunk_id", "")))
            next_frontier -= visited_nodes
            visited_nodes |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        cand_chunks = {c for c in cand_chunks if c}

        # Graph-score: share of nodes overlapping visited neighborhood
        scored: List[Tuple[str, float]] = []
        for cid in cand_chunks:
            nodes = self.chunk2nodes.get(cid, set())
            gscore = len(nodes & visited_nodes) / max(1, len(nodes))
            scored.append((cid, gscore))

        # Normalize vector scores among candidates
        vec_score_map = {cid: s for cid, s in vector_hits}
        vec_vals = [vec_score_map.get(cid, 0.0) for cid, _ in scored]
        vmin, vmax = (min(vec_vals), max(vec_vals)) if vec_vals else (0.0, 1.0)

        def norm(x: float) -> float:
            if vmax <= vmin:
                return 0.0
            return (x - vmin) / (vmax - vmin)

        merged: List[Tuple[str, float, float, float]] = []
        for cid, gscore in scored:
            v = norm(vec_score_map.get(cid, 0.0))
            final = w_vec * v + w_graph * gscore
            merged.append((cid, final, v, gscore))

        merged.sort(key=lambda x: x[1], reverse=True)

        # keep seeds first, then top expanded
        final_chunks: List[str] = []
        seen = set()
        for cid, _, _, _ in merged:
            if cid in seed_chunk_ids and cid not in seen:
                final_chunks.append(cid)
                seen.add(cid)
        for cid, _, _, _ in merged:
            if cid not in seen:
                final_chunks.append(cid)
                seen.add(cid)
            if len(final_chunks) >= top_seed + max_expand:
                break

        # Communities touched by visited_nodes
        comm_ids: Set[str] = set()
        for n in visited_nodes:
            comm_ids |= self.node2communities.get(n, set())

        comm_summaries = []
        for cid in sorted(list(comm_ids)):
            com = self.communities.get(cid)
            if com and com.summary:
                comm_summaries.append({"community_id": cid, "summary": com.summary})

        return {
            "seed_chunks": [{"chunk_id": cid, "score": float(s), "text": chunk_text_map.get(cid, "")} for cid, s in seeds],
            "expanded_chunks": [{"chunk_id": cid, "text": chunk_text_map.get(cid, "")} for cid in final_chunks if cid in chunk_text_map],
            "community_summaries": comm_summaries,
            "debug": {
                "seed_nodes": sorted(list(seed_nodes))[:50],
                "visited_nodes_size": len(visited_nodes),
                "candidate_chunks_size": len(cand_chunks),
                "communities_hit": sorted(list(comm_ids))[:50],
            },
        }


# -----------------------------
# Integration snippet (how to plug into your app)
# -----------------------------

def integrate_with_existing_rag(
    faiss_hits: List[Tuple[str, float]],
    chunk_text_map: Dict[str, str],
    graphrag_artifact_dir: str,
) -> str:
    """
    Example helper: return a single context string for prompt assembly.
    """
    retriever = GraphRAGRetriever(graphrag_artifact_dir)
    pack = retriever.expand(faiss_hits, chunk_text_map)
    parts: List[str] = []

    for s in pack["community_summaries"][:3]:
        parts.append(f"[CommunitySummary:{s['community_id']}]\n{s['summary']}")

    for item in pack["expanded_chunks"]:
        cid = item["chunk_id"]
        txt = (item["text"] or "").strip()
        if not txt:
            continue
        parts.append(f"[Chunk:{cid}]\n{txt}")

    return "\n\n".join(parts)
