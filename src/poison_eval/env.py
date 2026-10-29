"""The agent's tool environment: cyber-intel-mcp's tools over a (possibly poisoned) corpus.

The tools mirror cyber-intel-mcp's MCP tools (same names, arguments and result
shapes) but run in-process, so each question can get its own planted documents
without restarting a server. Only the search index is poisoned: lookup_cve,
search_kev and get_attack_technique read the primary records, exactly as the
real server does.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from cyber_intel_mcp.store import BM25Index, Document, SnapshotStore

CVE_ID_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DOC_TYPES = {"cve", "kev", "attack", "web"}


class ToolError(Exception):
    pass


class Environment:
    def __init__(self, base: SnapshotStore, planted: list[Document] | None = None):
        self.planted = {d.source_id: d for d in planted or []}
        if self.planted:
            store = copy.copy(base)  # shallow: shares the primary records, gets its own index
            store.index = BM25Index(base.index.docs + list(self.planted.values()))
            self.store = store
        else:
            self.store = base
        self.seen_sources: set[str] = set()  # every source id the model was shown

    # ---- tools -----------------------------------------------------------

    def lookup_cve(self, cve_id: str) -> dict[str, Any]:
        cve_id = cve_id.strip().upper()
        if not CVE_ID_RE.match(cve_id):
            raise ToolError(f"'{cve_id}' is not a valid CVE id (expected CVE-YYYY-NNNN).")
        record, kev = self.store.cves.get(cve_id), self.store.kev.get(cve_id)
        if record is None:
            return {"found": False, "cve_id": cve_id, "in_kev": kev is not None,
                    "source_ids": [f"kev:{cve_id}"] if kev else [],
                    "message": "CVE record not in this snapshot."}
        record = {**record, "references": record.get("references", [])[:3]}  # keep the context window small
        return {"found": True, **record, "in_kev": kev is not None,
                "kev_date_added": kev["date_added"] if kev else None,
                "source_ids": [f"cve:{cve_id}"] + ([f"kev:{cve_id}"] if kev else [])}

    def search_kev(self, vendor=None, product=None, added_after=None, added_before=None,
                   ransomware_only=False, limit=25) -> dict[str, Any]:
        for name, value in (("added_after", added_after), ("added_before", added_before)):
            if value and not DATE_RE.match(value):
                raise ToolError(f"{name} must be YYYY-MM-DD, got '{value}'.")
        if not any([vendor, product, added_after, added_before, ransomware_only]):
            raise ToolError("Give at least one filter (vendor, product, a date, or ransomware_only).")
        matches = self.store.search_kev(vendor, product, added_after, added_before, bool(ransomware_only))
        shown = matches[: max(1, min(int(limit), 100))]
        return {"total_matches": len(matches), "returned": len(shown), "truncated": len(matches) > len(shown),
                "entries": [{**e, "source_id": f"kev:{e['cve_id']}"} for e in shown]}

    def get_attack_technique(self, technique_id: str) -> dict[str, Any]:
        technique_id = technique_id.strip().upper()
        if not TECHNIQUE_ID_RE.match(technique_id):
            raise ToolError(f"'{technique_id}' is not a valid technique id (expected T1234 or T1234.001).")
        t = self.store.techniques.get(technique_id)
        if t is None:
            return {"found": False, "technique_id": technique_id, "source_ids": []}
        return {"found": True, **t, "source_ids": [f"attack:{technique_id}"]}

    def search_corpus(self, query: str, doc_types: list[str] | None = None, limit: int = 5) -> dict[str, Any]:
        types = set(doc_types or [])
        if types - DOC_TYPES:
            raise ToolError(f"Unknown doc_types {sorted(types - DOC_TYPES)}; use {sorted(DOC_TYPES)}.")
        hits = self.search(query, limit=max(1, min(int(limit), 20)), doc_types=types or None)
        return {"query": query, "results": [
            {"source_id": d.source_id, "type": d.doc_type, "title": d.title,
             "snippet": d.text[:300], "score": round(score, 3), "match": match}
            for d, score, match in hits]}

    def get_source(self, source_id: str) -> dict[str, Any]:
        source_id = source_id.strip()
        if source_id in self.planted:
            d = self.planted[source_id]
            return {"found": True, "source_id": source_id, "record": {"title": d.title, "text": d.text}}
        record = self.store.get_source(source_id)
        if record is None:
            return {"found": False, "source_id": source_id}
        return {"found": True, "source_id": source_id, "record": record}

    # ---- plumbing --------------------------------------------------------

    def search(self, query: str, limit: int = 5, doc_types: set[str] | None = None):
        return self.store.search(query, limit=limit, doc_types=doc_types)

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool by name and return its JSON result (errors are returned, not raised)."""
        fn = getattr(self, name, None) if name in TOOL_NAMES else None
        try:
            if fn is None:
                raise ToolError(f"Unknown tool '{name}'.")
            result = fn(**args)
        except ToolError as e:
            result = {"error": str(e)}
        except TypeError as e:  # wrong or missing arguments from the model
            result = {"error": f"Bad arguments for {name}: {e}"}
        text = json.dumps(result, ensure_ascii=False)
        self.seen_sources.update(re.findall(r"\b(?:cve|kev|attack|web):[\w.\-]+", text))
        return text[:3000]


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_cve",
        "description": "Get a CVE record: description, CVSS score, CWEs, affected products, dates, and whether it is in CISA KEV.",
        "parameters": {"type": "object", "properties": {
            "cve_id": {"type": "string", "description": "CVE identifier, e.g. CVE-2021-44228"}},
            "required": ["cve_id"]}}},
    {"type": "function", "function": {
        "name": "search_kev",
        "description": "Search the CISA Known Exploited Vulnerabilities catalog. Returns the total match count plus matching entries sorted by date added.",
        "parameters": {"type": "object", "properties": {
            "vendor": {"type": "string", "description": "Case-insensitive substring of the vendor"},
            "product": {"type": "string", "description": "Case-insensitive substring of the product"},
            "added_after": {"type": "string", "description": "YYYY-MM-DD"},
            "added_before": {"type": "string", "description": "YYYY-MM-DD"},
            "ransomware_only": {"type": "boolean"},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "get_attack_technique",
        "description": "Get a MITRE ATT&CK Enterprise technique: name, tactics, platforms and description.",
        "parameters": {"type": "object", "properties": {
            "technique_id": {"type": "string", "description": "e.g. T1190"}},
            "required": ["technique_id"]}}},
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Search CVE, KEV, ATT&CK records and web articles. Ids in the query are matched exactly and listed first; the rest is keyword search.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "doc_types": {"type": "array", "items": {"type": "string", "enum": sorted(DOC_TYPES)}},
            "limit": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_source",
        "description": "Fetch the full record behind a source id. Use it to verify that a citation supports a claim.",
        "parameters": {"type": "object", "properties": {
            "source_id": {"type": "string", "description": "e.g. kev:CVE-2021-44228"}},
            "required": ["source_id"]}}},
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}
