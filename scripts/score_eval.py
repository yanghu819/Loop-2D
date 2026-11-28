"""Aggregate multi-rank evaluation shards and compute exact problem-paired Delta +/- SE & z-scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path


def load_arm_records(pattern_or_dir: str) -> dict[str, list[dict]]:
    p = Path(pattern_or_dir)
    if p.is_dir():
        files = sorted(str(x) for x in p.glob("rank*.json") if not x.name.endswith(".partial.json"))
        if not files:
            files = sorted(str(x) for x in p.glob("*.json") if not x.name.endswith(".partial.json"))
    elif p.is_file():
        files = [str(p)]
    else:
        files = sorted(glob.glob(pattern_or_dir))
    if not files:
        raise FileNotFoundError(f"No eval JSON files matched: {pattern_or_dir}")

    by_problem: dict[str, list[dict]] = defaultdict(list)
    for fp in files:
        d = json.loads(Path(fp).read_text())
        for r in (d if isinstance(d, list) else d.get("records", [])):
            by_problem[r["id"]].append(r)
    return dict(by_problem)


def summarize_arm(by_prob: dict[str, list[dict]]) -> dict:
    by_bench: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    for pid, recs in by_prob.items():
        bench = recs[0]["bench"]
        acc = sum(1.0 if r["correct"] else 0.0 for r in recs) / len(recs)
        avg_len = sum(r["gen_tokens"] for r in recs) / len(recs)
        trunc = sum(0.0 if r["finished"] else 1.0 for r in recs) / len(recs)
        by_bench[bench].append((pid, acc, avg_len, trunc))

    out: dict = {"benchmarks": {}, "problem_acc": {}}
    for b, rows in sorted(by_bench.items()):
        n = len(rows)
        mean_acc = sum(x[1] for x in rows) / n
        mean_len = sum(x[2] for x in rows) / n
        mean_trunc = sum(x[3] for x in rows) / n
        var = sum((x[1] - mean_acc) ** 2 for x in rows) / max(1, n - 1)
        out["benchmarks"][b] = {
            "n_problems": n,
            "samples_per_prob": len(by_prob[rows[0][0]]),
            "acc_pct": round(mean_acc * 100.0, 2),
            "se_pct": round(math.sqrt(var / n) * 100.0, 2),
            "mean_len_tok": round(mean_len, 1),
            "trunc_pct": round(mean_trunc * 100.0, 2),
        }
        for pid, acc, _, _ in rows:
            out["problem_acc"][pid] = (b, acc)

    comp_benches = [b for b in ("aime24", "aime25", "aime26", "amc23") if b in out["benchmarks"]]
    if comp_benches:
        comp_rows = [r for b in comp_benches for r in by_bench[b]]
        out["competition_pooled_pct"] = round(sum(x[1] for x in comp_rows) / len(comp_rows) * 100.0, 2)
        out["competition_macro_pct"] = round(
            sum(out["benchmarks"][b]["acc_pct"] for b in comp_benches) / len(comp_benches), 2
        )
    return out


def paired_compare(arm_a: dict, arm_b: dict, bench_filter: set[str] | None = None) -> dict:
    pa, pb = arm_a["problem_acc"], arm_b["problem_acc"]
    common = sorted(k for k in pa if k in pb and (bench_filter is None or pa[k][0] in bench_filter))
    if not common:
        return {"n_problems": 0}
    diffs = [(pa[k][1] - pb[k][1]) * 100.0 for k in common]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(1, n - 1)
    se_d = math.sqrt(var_d / n)
    z = mean_d / se_d if se_d > 1e-12 else 0.0
    return {"n_problems": n, "delta_pct": round(mean_d, 2), "se_pct": round(se_d, 2), "z": round(z, 2)}


def main() -> None:
    p = argparse.ArgumentParser(description="Score and pair-compare 2D-AttnRes benchmark runs")
    p.add_argument("--arms", nargs="+", required=True, help="name=path_or_glob pairs")
    p.add_argument("--ref", default=None, help="Reference arm name for paired Delta +/- SE")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    summaries = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        summaries[name] = summarize_arm(load_arm_records(path))

    ref_name = args.ref or next(iter(summaries))
    ref_sum = summaries[ref_name]
    report = {"ref": ref_name, "arms": {}, "paired_vs_ref": {}}

    print(f"\n=== 2D-AttnRes EVAL SUMMARY (ref={ref_name}) ===")
    for name, s in summaries.items():
        report["arms"][name] = {k: v for k, v in s.items() if k != "problem_acc"}
        b_str = " | ".join(f"{b}={v['acc_pct']:.2f}% ({v['mean_len_tok']:.0f}t)" for b, v in s["benchmarks"].items())
        comp_s = f"CompPooled={s.get('competition_pooled_pct', 0):.2f}%" if "competition_pooled_pct" in s else ""
        print(f"  {name:24s} | {comp_s:20s} | {b_str}")
        if name != ref_name:
            comp_cmp = paired_compare(s, ref_sum, {"aime24", "aime25", "aime26", "amc23"})
            m500_cmp = paired_compare(s, ref_sum, {"math500"})
            gpqa_cmp = paired_compare(s, ref_sum, {"gpqa_diamond"})
            report["paired_vs_ref"][name] = {"competition": comp_cmp, "math500": m500_cmp, "gpqa_diamond": gpqa_cmp}
            parts = []
            for label, cmp_d in (("Comp", comp_cmp), ("MATH500", m500_cmp), ("GPQA", gpqa_cmp)):
                if cmp_d.get("n_problems", 0) > 0:
                    parts.append(f"{label} Δ={cmp_d['delta_pct']:+.2f}±{cmp_d['se_pct']:.2f}% (z={cmp_d['z']:+.2f})")
            if parts:
                print(f"    vs {ref_name:16s} | " + " | ".join(parts))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"SAVED {args.out}")


if __name__ == "__main__":
    main()
