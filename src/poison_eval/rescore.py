"""Re-score saved runs after a scoring or parsing fix, without calling the model again.

    poison-rescore

What it does to results/runs.jsonl (a backup copy is saved first):
  - re-parses every saved response and recomputes outcome, sources and cited_web
  - removes dry-run rows from the fake model
  - removes runs made with an older version of a condition's design (see
    CONDITION_VERSIONS in agents.py), so the next `poison-run` redoes only those
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from .agents import CONDITION_VERSIONS, parse_answer
from .questions import DATA_DIR, load_questions
from .run import RESULTS_DIR, _version
from .scoring import score


def rescore(runs: list[dict], questions: dict, keep_fake: bool = False, keep_old_rag: bool = False):
    kept, dropped = [], Counter()
    for r in runs:
        if r.get("model") == "fake" and not keep_fake:
            dropped["fake dry-run rows"] += 1
            continue
        if _version(r) != CONDITION_VERSIONS.get(r.get("condition"), 1) and not keep_old_rag:
            dropped[f"outdated {r.get('condition')} rows (will be rerun)"] += 1
            continue
        if r.get("outcome") == "error" or "response" not in r:
            kept.append(r)
            continue
        q = questions[r["qid"]]
        final, sources = parse_answer(r["response"])
        r.update(final_answer=final, sources=sources, outcome=score(q, final),
                 cited_web=any(s.lower().startswith("web:") for s in sources))
        if r.get("draft_response") is not None:
            r["draft_answer"], _ = parse_answer(r["draft_response"])
            r["draft_outcome"] = score(q, r["draft_answer"])
        elif r.get("condition") == "agent_verify":
            r["draft_outcome"] = None  # draft text wasn't saved by older versions
        kept.append(r)
    return kept, dropped


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Re-score saved runs without rerunning the model.")
    ap.add_argument("--runs", type=Path, default=RESULTS_DIR / "runs.jsonl")
    ap.add_argument("--questions", type=Path, default=DATA_DIR / "questions.jsonl")
    ap.add_argument("--keep-fake", action="store_true")
    ap.add_argument("--keep-old-rag", "--keep-outdated", action="store_true")
    args = ap.parse_args(argv)

    with open(args.runs, encoding="utf-8") as f:
        runs = [json.loads(l) for l in f if l.strip()]
    before = Counter(r.get("outcome") for r in runs if r.get("model") != "fake")
    backup = args.runs.with_name(f"runs.backup-{datetime.now():%Y%m%d-%H%M%S}.jsonl")
    shutil.copyfile(args.runs, backup)

    questions = {q.qid: q for q in load_questions(args.questions)}
    kept, dropped = rescore(runs, questions, args.keep_fake, args.keep_old_rag)
    with open(args.runs, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"Backup saved: {backup}")
    print(f"Outcomes before: {dict(before)}")
    print(f"Outcomes after:  {dict(Counter(r.get('outcome') for r in kept))}")
    for what, n in dropped.items():
        print(f"Removed {n} {what}")
    print(f"{len(kept)} runs kept in {args.runs}")


if __name__ == "__main__":
    main()
