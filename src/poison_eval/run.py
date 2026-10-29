"""Run the experiment: every question x poison level x condition, one JSON line per run.

Resumable: runs already in the output file are skipped, so you can stop with
Ctrl+C and pick up later by running the same command.

    poison-run --limit 8                      # quick pilot
    poison-run                                # full run (60 questions x 3 levels x 3 conditions)
    poison-run --model llama3.1:8b            # another local model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .agents import CONDITION_VERSIONS, CONDITIONS
from .env import Environment
from .llm import OLLAMA_URL, ChatClient, FakeLLM, LLMError
from .poison import MAX_POISON, make_poison_docs
from .questions import DATA_DIR, load_questions, load_store
from .scoring import score

RESULTS_DIR = DATA_DIR.parent / "results"


def _version(r: dict) -> int:
    """Design version a saved run was made with (older files used rag_context_version)."""
    return r.get("condition_version") or r.get("rag_context_version") or 1


def run_one(llm, base_store, q, level: int, condition: str) -> dict:
    env = Environment(base_store, make_poison_docs(q, level))
    record = {"qid": q.qid, "qtype": q.qtype, "cve_id": q.cve_id, "condition": condition,
              "poison_level": level, "model": llm.model, "gold": q.gold, "poison": q.poison}
    try:
        r = CONDITIONS[condition](llm, env, q)
    except LLMError:
        raise
    except Exception as e:  # one bad run should not stop the experiment
        return {**record, "outcome": "error", "error": f"{type(e).__name__}: {e}"}
    outcome = score(q, r.final_answer)
    record.update({
        "outcome": outcome,
        "final_answer": r.final_answer,
        "draft_answer": r.draft_answer,
        "draft_outcome": score(q, r.draft_answer) if r.draft_answer is not None else None,
        "sources": r.sources,
        "cited_web": any(s.lower().startswith("web:") for s in r.sources),
        "exposed": any(s.startswith("web:") for s in env.seen_sources),
        "tool_calls": r.tool_calls,
        "tool_args": r.tool_args,
        "n_tool_calls": len(r.tool_calls),
        "llm_calls": r.llm_calls,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "latency_s": round(r.latency_s, 2),
        "response": r.text[-2000:],
        "draft_response": r.draft_text[-2000:] if r.draft_text is not None else None,
        "condition_version": CONDITION_VERSIONS.get(condition, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the narrative-poisoning experiment.")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--base-url", default=OLLAMA_URL, help="OpenAI-compatible endpoint (default: local Ollama)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--poison", nargs="+", type=int, default=[0, 1, MAX_POISON], help="poison levels")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--qtypes", nargs="+", default=None, help="only these question types")
    ap.add_argument("--questions", type=Path, default=DATA_DIR / "questions.jsonl")
    ap.add_argument("--data-dir", type=Path, default=None, help="snapshot dir (default: bundled sample)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "runs.jsonl")
    ap.add_argument("--fake", action="store_true", help="dry run with a fake model (tests the pipeline)")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    if args.qtypes:
        questions = [q for q in questions if q.qtype in args.qtypes]
    questions = questions[: args.limit] if args.limit else questions

    if args.fake:
        llm = FakeLLM(lambda m, t: {"content": "FINAL ANSWER: unknown\nSOURCES:"})
    else:
        llm = ChatClient(args.model, args.base_url)
    try:
        llm.check()
    except LLMError as e:
        sys.exit(f"Error: {e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("outcome") != "error" and \
                            _version(r) == CONDITION_VERSIONS.get(r["condition"], 1):
                        done.add((r["qid"], r["condition"], r["poison_level"], r["model"]))

    base_store = load_store(args.data_dir)
    jobs = [(q, lvl, c) for q in questions for lvl in args.poison for c in args.conditions
            if (q.qid, c, lvl, llm.model) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already in {args.out}); model {llm.model}")

    start = time.perf_counter()
    with open(args.out, "a", encoding="utf-8") as out:
        for i, (q, level, cond) in enumerate(jobs, 1):
            try:
                rec = run_one(llm, base_store, q, level, cond)
            except LLMError as e:
                sys.exit(f"\nStopped: {e}\nProgress is saved; rerun the same command to continue.")
            out.write(json.dumps(rec) + "\n")
            out.flush()
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {q.qid} {q.qtype:<10} poison={level} {cond:<12} -> "
                  f"{rec['outcome']:<8} {rec.get('final_answer', rec.get('error', ''))[:40]!r} "
                  f"(eta {eta / 60:.0f} min)", flush=True)
    print(f"Done. Results in {args.out}. Next: poison-summarize")


if __name__ == "__main__":
    main()
