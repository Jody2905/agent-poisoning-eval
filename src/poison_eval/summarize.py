"""Summarize results/runs.jsonl into tables (and a chart if matplotlib is installed).

Metrics per model x condition x poison level:
  accuracy        share of answers matching the snapshot
  ASR             attack success rate: share that repeated the planted false claim
  ASR if exposed  ASR among runs where the model actually saw a planted document
  cited web       share of answers citing a web: source
Brackets are 95% Wilson confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .run import RESULTS_DIR


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int, ci: bool = True) -> str:
    if n == 0:
        return "-"
    s = f"{100 * k / n:.0f}%"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.0f}-{100 * hi:.0f}]"
    return s


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def summarize(runs: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("outcome") != "error":
            groups[tuple(r[k] for k in group_keys)].append(r)
    rows = []
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        exposed = [r for r in g if r["exposed"]]
        rows.append({
            **dict(zip(group_keys, key)),
            "n": n,
            "correct": sum(r["outcome"] == "correct" for r in g),
            "poisoned": sum(r["outcome"] == "poisoned" for r in g),
            "n_exposed": len(exposed),
            "poisoned_exposed": sum(r["outcome"] == "poisoned" for r in exposed),
            "cited_web": sum(r["cited_web"] for r in g),
            "mean_latency_s": round(sum(r["latency_s"] for r in g) / n, 1),
            "mean_tool_calls": round(sum(r["n_tool_calls"] for r in g) / n, 1),
            "mean_tokens": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in g) / n),
        })
    return rows


def to_markdown(rows: list[dict], keys: tuple[str, ...]) -> str:
    head = [*keys, "n", "accuracy", "ASR", "ASR if exposed", "cited web", "latency (s)", "tool calls", "tokens"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join([
            *[str(r[k]) for k in keys], str(r["n"]),
            pct(r["correct"], r["n"]), pct(r["poisoned"], r["n"]),
            pct(r["poisoned_exposed"], r["n_exposed"]), pct(r["cited_web"], r["n"], ci=False),
            str(r["mean_latency_s"]), str(r["mean_tool_calls"]), str(r["mean_tokens"]),
        ]) + " |")
    return "\n".join(lines)


def chart(rows: list[dict], path: Path, model: str = "") -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    order = ["rag", "agent", "agent_verify", "agent_search", "agent_search_verify"]
    conditions = sorted({r["condition"] for r in rows}, key=lambda c: order.index(c) if c in order else 99)
    levels = sorted({r["poison_level"] for r in rows if r["poison_level"] > 0})
    if not levels:
        return False
    fig, ax = plt.subplots(figsize=(1.8 * len(conditions) + 2, 4))
    width = 0.8 / len(levels)
    for i, lvl in enumerate(levels):
        vals, errs = [], [[], []]
        for c in conditions:
            r = next((r for r in rows if r["condition"] == c and r["poison_level"] == lvl), None)
            k, n = (r["poisoned"], r["n"]) if r else (0, 0)
            p = k / n if n else 0
            lo, hi = wilson(k, n)
            vals.append(100 * p)
            errs[0].append(100 * (p - lo))
            errs[1].append(100 * (hi - p))
        xs = [j + (i - (len(levels) - 1) / 2) * width for j in range(len(conditions))]
        ax.bar(xs, vals, width, yerr=errs, capsize=3, label=f"{lvl} planted doc{'s' if lvl > 1 else ''}")
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Answers that repeated the planted claim ({model})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=200)
    except OSError as e:  # e.g. the old chart is open in an image viewer on Windows
        print(f"Could not save {path}: {e}. Close the image if it is open, then run poison-summarize again.")
        return False
    finally:
        plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Summarize experiment results.")
    ap.add_argument("--runs", type=Path, default=RESULTS_DIR / "runs.jsonl")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    from .agents import CONDITION_VERSIONS
    from .run import _version
    all_runs = load(args.runs)
    runs = [r for r in all_runs if _version(r) == CONDITION_VERSIONS.get(r.get("condition"), 1)]
    if len(runs) < len(all_runs):
        print(f"Skipping {len(all_runs) - len(runs)} runs from older designs (poison-rescore removes them).")
    errors = sum(r.get("outcome") == "error" for r in runs)
    main_keys = ("model", "condition", "poison_level")
    type_keys = ("model", "condition", "poison_level", "qtype")
    main_rows, type_rows = summarize(runs, main_keys), summarize(runs, type_keys)

    md = ["# Results", "", f"{len(runs) - errors} scored runs" + (f", {errors} errors (excluded)" if errors else ""), "",
          "## By condition and poison level", "", to_markdown(main_rows, main_keys), "",
          "## By question type", "", to_markdown(type_rows, type_keys), ""]
    verify = [r for r in runs if r.get("condition") == "agent_verify" and r.get("draft_outcome")]
    if verify:
        fixed = sum(r["draft_outcome"] == "poisoned" and r["outcome"] == "correct" for r in verify)
        broke = sum(r["draft_outcome"] == "correct" and r["outcome"] != "correct" for r in verify)
        md += ["## What verification changed", "",
               f"- Poisoned drafts corrected by verification: {fixed}",
               f"- Correct drafts made wrong by verification: {broke}", ""]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    with open(args.out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(main_rows[0]) if main_rows else ["n"])
        w.writeheader()
        w.writerows(main_rows)
    print("\n".join(md))
    for model in sorted({r["model"] for r in main_rows}):
        path = args.out_dir / f"asr_{model.replace(':', '-').replace('/', '-')}.png"
        if chart([r for r in main_rows if r["model"] == model], path, model):
            print(f"Chart: {path}")


if __name__ == "__main__":
    main()
