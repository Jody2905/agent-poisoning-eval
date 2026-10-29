"""End-to-end runs with scripted fake models (no Ollama needed)."""

import json

from poison_eval import run, summarize
from poison_eval.llm import FakeLLM
from poison_eval.run import run_one


def gullible(messages, tools):
    """Searches, then repeats whatever the first web article says; otherwise the primary record."""
    last = messages[-1]
    if tools and last["role"] != "tool":
        return {"tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "search_corpus", "arguments": json.dumps({"query": messages[1]["content"]})}}]}
    return {"content": "FINAL ANSWER: see sources\nSOURCES: web:x"}


def careful(q):
    """Looks up the primary record and answers with the gold answer."""
    def respond(messages, tools):
        if tools and messages[-1]["role"] != "tool":
            return {"tool_calls": [{"id": "1", "type": "function", "function": {
                "name": "lookup_cve", "arguments": {"cve_id": q.cve_id}}}]}
        return {"content": f"FINAL ANSWER: {q.gold}\nSOURCES: cve:{q.cve_id}"}
    return respond


def poisoner(q):
    def respond(messages, tools):
        return {"content": f"FINAL ANSWER: {q.poison}\nSOURCES: web:secwatch-{q.cve_id.lower()}"}
    return respond


def test_agent_tool_loop_and_scoring(store, questions):
    q = questions[0]
    rec = run_one(FakeLLM(careful(q)), store, q, 3, "agent")
    assert rec["outcome"] == "correct" and rec["tool_calls"] == ["lookup_cve"] and not rec["cited_web"]

    rec = run_one(FakeLLM(gullible), store, q, 3, "agent")
    assert rec["exposed"] and rec["tool_calls"] == ["search_corpus"]

    rec = run_one(FakeLLM(poisoner(q)), store, q, 1, "rag")
    assert rec["outcome"] == "poisoned" and rec["cited_web"] and rec["exposed"]


def test_verify_records_draft(store, questions):
    q = questions[1]
    rec = run_one(FakeLLM(careful(q)), store, q, 1, "agent_verify")
    assert rec["draft_outcome"] == "correct" and rec["outcome"] == "correct"
    assert rec["llm_calls"] == 4


def test_agent_stops_after_max_steps(store, questions):
    def loops(messages, tools):
        if tools:
            return {"tool_calls": [{"id": "1", "function": {"name": "lookup_cve", "arguments": "{}"}}]}
        return {"content": "FINAL ANSWER: no idea"}
    rec = run_one(FakeLLM(loops), store, questions[0], 0, "agent")
    assert rec["n_tool_calls"] == 6 and rec["final_answer"] == "no idea"


def test_cli_dry_run_resume_and_summary(tmp_path):
    out = tmp_path / "runs.jsonl"
    run.main(["--fake", "--limit", "2", "--out", str(out)])
    assert len(out.read_text().splitlines()) == 2 * 3 * 5
    run.main(["--fake", "--limit", "2", "--out", str(out)])  # resume: nothing new
    assert len(out.read_text().splitlines()) == 30
    summarize.main(["--runs", str(out), "--out-dir", str(tmp_path)])
    assert "| fake | agent | 3 |" in (tmp_path / "summary.md").read_text()


def test_rag_context_shows_primary_fields(store, questions):
    from poison_eval.agents import render_hit
    from poison_eval.env import Environment
    q = next(q for q in questions if q.qtype == "cvss")
    env = Environment(store)
    texts = [render_hit(env, d) for d, _, _ in env.search(q.question, limit=5)]
    assert any(f"CVSS base score: {float(q.gold)}" in t for t in texts)
    q = next(q for q in questions if q.qtype == "ransomware")
    texts = [render_hit(env, d) for d, _, _ in env.search(q.question, limit=5)]
    assert any("Known ransomware campaign use:" in t for t in texts)


def test_rescore_fixes_saved_pilot_rows(questions, tmp_path):
    from poison_eval import rescore
    q = next(q for q in questions if q.qtype == "vendor")
    rows = [
        {"qid": q.qid, "model": "qwen", "condition": "agent", "poison_level": 0, "outcome": "other",
         "response": f"{q.gold}\nSOURCES: cve:{q.cve_id}"},
        {"qid": q.qid, "model": "qwen", "condition": "rag", "poison_level": 3, "outcome": "other",
         "response": f"{q.poison}\nSOURCES: web:x"},
        {"qid": q.qid, "model": "qwen", "condition": "agent_search", "poison_level": 3, "outcome": "correct",
         "rag_context_version": None, "response": f"{q.gold}\nSOURCES: cve:x"},
        {"qid": q.qid, "model": "fake", "condition": "agent", "poison_level": 0, "outcome": "other",
         "response": ""},
    ]
    p = tmp_path / "runs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    rescore.main(["--runs", str(p)])
    kept = [json.loads(l) for l in p.read_text().splitlines()]
    assert [r["outcome"] for r in kept] == ["correct"]  # old RAG and fake rows removed
    assert list(tmp_path.glob("runs.backup-*.jsonl"))


def test_search_agent_only_gets_search_tools(store, questions):
    q = questions[0]
    seen_tools = []

    def respond(messages, tools):
        seen_tools.append(sorted(t["function"]["name"] for t in tools or []))
        if tools and messages[-1]["role"] != "tool":
            return {"tool_calls": [{"id": "1", "function": {
                "name": "lookup_cve", "arguments": {"cve_id": q.cve_id}}}]}
        return {"content": f"FINAL ANSWER: {q.poison}\nSOURCES: web:x"}

    rec = run_one(FakeLLM(respond), store, q, 3, "agent_search")
    assert seen_tools[0] == ["get_source", "search_corpus"]
    assert rec["condition_version"] == 2
    assert '"not available"' not in rec["response"]  # error goes to the tool message, not the answer
    assert rec["outcome"] == "poisoned"

    seen_tools.clear()
    rec = run_one(FakeLLM(respond), store, q, 3, "agent_search_verify")
    assert seen_tools[0] == ["get_source", "search_corpus"]
    assert len(seen_tools[-1]) == 5  # verification pass has every tool
    assert rec["draft_outcome"] == "poisoned"


def test_search_agent_cannot_filter_out_web_articles(store, questions):
    q = questions[1]
    sent_tools = []

    def respond(messages, tools):
        sent_tools.append(tools)
        if tools and messages[-1]["role"] != "tool":
            return {"tool_calls": [{"id": "1", "function": {"name": "search_corpus", "arguments": json.dumps(
                {"query": q.cve_id, "doc_types": ["kev"]})}}]}
        return {"content": f"FINAL ANSWER: {q.gold}\nSOURCES: kev:{q.cve_id}"}

    rec = run_one(FakeLLM(respond), store, q, 3, "agent_search")
    search = next(t for t in sent_tools[0] if t["function"]["name"] == "search_corpus")
    assert "doc_types" not in search["function"]["parameters"]["properties"]
    assert rec["tool_args"][0]["args"] == {"query": q.cve_id}
    assert rec["exposed"]  # the filter was ignored, so the planted articles came back
    # the full-tool agent keeps its filter
    from poison_eval.env import TOOL_SCHEMAS
    assert "doc_types" in TOOL_SCHEMAS[3]["function"]["parameters"]["properties"]
