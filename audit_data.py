"""Audit saved measurements without loading models or altering original results.

Run with the project's Python environment: python audit_data.py
Outputs: analysis/audit.json, analysis/prompt_metrics.csv and DATA_REVIEW.md.
The ten translated question IDs, not the repeated trials, are the comparison units.
"""
import csv
import hashlib
import itertools
import json
import math
import random
import statistics as st
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = ("en", "hi", "ur")
KS = (2, 4, 6, 8)


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("repeat", "prompt_id", "k", "tokens", "passes"):
            r[key] = int(r[key])
        for key in ("sec", "tps", "tau", "wall", "script_ratio", "tokens_per_word", "chars_per_token"):
            r[key] = float(r[key])
        r["sm_mhz"] = int(r["sm_mhz"]) if r.get("sm_mhz") else None
        for key in ("matches_baseline", "lang_ok"):
            if r[key] not in ("True", "False"):
                raise ValueError(f"Invalid Boolean in {key}: {r[key]}")
            r[key] = r[key] == "True"
    return rows


def interval(values, seed=0):
    """Descriptive paired-prompt bootstrap; not a population confidence claim."""
    rng = random.Random(seed)
    draws = sorted(st.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [round(draws[249], 4), round(draws[9749], 4)]


def prompt_values(rows, lang, k, field, predicate=lambda r: True):
    selected = [r for r in rows if r["lang"] == lang and r["k"] == k and predicate(r)]
    return {p: st.median(r[field] for r in selected if r["prompt_id"] == p)
            for p in sorted({r["prompt_id"] for r in selected})}


def speed(rows, lang, k, predicate=lambda r: True):
    base = prompt_values(rows, lang, 0, "tps")
    spec = prompt_values(rows, lang, k, "tps", predicate)
    values = [spec[p] / base[p] for p in sorted(base.keys() & spec.keys())]
    return {"n_prompts": len(values), "median": round(st.median(values), 6) if values else None,
            "bootstrap_median_interval": interval(values) if values else None,
            "prompts_above_one": sum(x > 1 for x in values)}


def contrast(rows, k, lower, upper, predicate=lambda r: True):
    a = prompt_values(rows, lower, k, "tau", predicate)
    b = prompt_values(rows, upper, k, "tau", predicate)
    ps = sorted(a.keys() & b.keys())
    differences = [a[p] - b[p] for p in ps]
    nneg, npos = sum(x < 0 for x in differences), sum(x > 0 for x in differences)
    n = nneg + npos
    # Exact sign test is descriptive only: curated questions are not a random sample.
    p = min(1, 2 * sum(math.comb(n, j) for j in range(min(nneg, npos) + 1)) / 2**n) if n else 1
    return {"n_pairs": len(ps), "lower_count": nneg, "upper_count": npos,
            "ties": len(ps) - n, "median_difference": round(st.median(differences), 4) if ps else None,
            "bootstrap_difference_interval": interval(differences) if ps else None,
            "descriptive_sign_p": round(p, 6)}


def audit(folder):
    rows = load_rows(folder / "raw.csv")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    outputs = json.loads((folder / "outputs.json").read_text(encoding="utf-8"))
    counts = Counter((r["lang"], r["prompt_id"], r["k"], r["repeat"]) for r in rows)
    expected = set(itertools.product(LANGS, range(10), (0,) + KS, range(2)))
    errors = []
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        errors.append("Missing, extra or duplicated condition keys")
    for i, r in enumerate(rows):
        if r["tokens"] <= 0 or r["sec"] <= 0 or r["passes"] <= 0:
            errors.append(f"Invalid count/time at row {i}")
        if abs(r["tau"] - r["tokens"] / r["passes"]) > 0.00051:
            errors.append(f"tau arithmetic mismatch at row {i}")
        # sec is saved to 0.001 s and tps to 0.01, so permit quantization error.
        if abs(r["tps"] - r["tokens"] / r["sec"]) > 0.0051 + r["tokens"] * 0.00051 / r["sec"]**2:
            errors.append(f"throughput arithmetic mismatch at row {i}")
        if r["k"] == 0 and (r["passes"] != r["tokens"] or not r["matches_baseline"]):
            errors.append(f"Baseline self-consistency failed at row {i}")
        if r["k"] and not (1 <= r["tau"] <= r["k"] + 1):
            errors.append(f"Invalid speculative tokens/pass bound at row {i}")
    outkeys = {(o["lang"], o["prompt_id"]) for o in outputs}
    if outkeys != set(itertools.product(LANGS, range(10))) or len(outputs) != 30:
        errors.append("Canonical output coverage mismatch")
    by_lang = {}
    prompt_table = []
    all_errors = []
    for lang in LANGS:
        lr = [r for r in rows if r["lang"] == lang]
        base = prompt_values(rows, lang, 0, "tps")
        clocks = [r["sm_mhz"] for r in lr if r["sm_mhz"] is not None]
        e = {"base_tps": round(st.median(base.values()), 4),
             "baseline_range": [min(r["tps"] for r in lr if r["k"] == 0), max(r["tps"] for r in lr if r["k"] == 0)],
             "clock_range": [min(clocks), max(clocks)],
             "clock_samples_below_80pct_max": sum(x < 0.8 * max(clocks) for x in clocks),
             "canonical_script_ratio_min": min(o["script_ratio"] for o in outputs if o["lang"] == lang),
             "canonical_chars_median": st.median(len(o["output"]) for o in outputs if o["lang"] == lang),
             "baseline_shorter_than_128": sum(r["tokens"] < 128 for r in lr if r["k"] == 0), "by_k": {}}
        for k in KS:
            kr = [r for r in lr if r["k"] == k]
            pooled = sum(r["tokens"] for r in kr) / sum(r["passes"] for r in kr)
            s = speed(rows, lang, k)
            prediction = pooled / (1 + k * summary["c"])
            error = prediction / s["median"] - 1
            all_errors.append(abs(error))
            s.update(tau_pooled=round(pooled, 4), predicted=round(prediction, 4),
                     prediction_relative_error=round(error, 4),
                     exact_matches=sum(r["matches_baseline"] for r in kr), n_runs=len(kr),
                     repeat_speedups=[speed([r for r in rows if r["repeat"] == rep], lang, k)["median"] for rep in (0, 1)],
                     matching_only=speed(rows, lang, k, lambda r: r["matches_baseline"]),
                     full_length_only=speed(rows, lang, k, lambda r: r["tokens"] == 128 and all(
                         b["tokens"] == 128 for b in rows if b["lang"] == lang and b["prompt_id"] == r["prompt_id"] and b["k"] == 0)))
            e["by_k"][str(k)] = s
            saved = summary["langs"][lang]["by_k"][str(k)]
            if abs(saved["speedup"] - s["median"]) > 0.0006 or abs(saved["tau_pooled"] - pooled) > 0.00051:
                errors.append(f"Saved summary mismatch: {lang}, K={k}")
            taus = prompt_values(rows, lang, k, "tau")
            tps = prompt_values(rows, lang, k, "tps")
            for p in sorted(base):
                pr = [r for r in kr if r["prompt_id"] == p]
                prompt_table.append(dict(dataset=folder.name, lang=lang, prompt_id=p, k=k,
                                         speedup=tps[p] / base[p], tau=taus[p],
                                         exact_matches=sum(r["matches_baseline"] for r in pr), repeats=len(pr)))
        by_lang[lang] = e
    spec = [r for r in rows if r["k"]]
    contrasts = {str(k): {f"{a}_vs_{b}": contrast(rows, k, a, b) for a, b in (("ur", "hi"), ("ur", "en"), ("hi", "en"))} for k in KS}
    matched_contrasts = {str(k): {f"{a}_vs_{b}": contrast(rows, k, a, b, lambda r: r["matches_baseline"]) for a, b in (("ur", "hi"), ("ur", "en"))} for k in KS}
    hashes = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in ("raw.csv", "outputs.json", "summary.json")}
    return dict(dataset=folder.name, n_runs=len(rows), errors=errors, hashes=hashes,
                c=summary["c"], exact_matches=sum(r["matches_baseline"] for r in spec), n_spec=len(spec),
                prediction_abs_error_max=round(max(all_errors), 4),
                langs=by_lang, paired_tau=contrasts, matching_only_paired_tau=matched_contrasts), prompt_table


def write_review(audits):
    final = audits["results_rp105"]
    lines = ["# Data review before the presentation", "", "Reviewed 12 October 2026. Reproduce the saved-data audit with `python audit_data.py`. No new GPU timing sweep was made; a separate targeted correctness probe is described below if present.", "",
             "## Verdict", "", "The saved local data supports a small, descriptive benchmark: no tested K improved median throughput; Urdu had lower tokens per target pass than Hindi on all ten translated questions. Hindi and English were similar on this metric. Exact output equivalence was not achieved on every run. The historical outputs alone cannot diagnose all mismatches; the targeted probe below supplies additional evidence for one case.", "",
             "## Integrity and provenance", ""]
    for name, a in audits.items():
        lines.append(f"- `{name}`: {a['n_runs']} rows; {len(a['errors'])} integrity/summary errors; {a['exact_matches']}/{a['n_spec']} speculative runs exactly matched the baseline.")
    lines += ["- Each dataset contains exactly 10 questions × 3 languages × 5 conditions × 2 repeats. Baseline passes equal generated tokens and both baseline repeats match their canonical baseline.",
              "- Raw CSVs and historical summaries were preserved. SHA-256 fingerprints are in `analysis/audit.json`.",
              "- Local notebook saved outputs support the final run. Both Kaggle notebooks have no saved executions; no T4 measurements or third hardware configuration are present. Count 600 recorded sweep trials, not 627 verified trials.",
              "- Earlier `results/summary.json` omits repetition penalty. The 1.0 attribution comes from project notes, not a complete saved run manifest.", "",
              "## Final run: what to put on the slides", "",
              "| Language | Baseline tok/s | Best tested K | Median speedup | Pooled tokens/target pass at K=2 | Exact matches at K=2 |", "|---|---:|---:|---:|---:|---:|"]
    for lang in LANGS:
        e = final["langs"][lang]; d = e["by_k"]["2"]
        lines.append(f"| {lang} | {e['base_tps']:.1f} | 2 | {d['median']:.2f}× | {d['tau_pooled']:.3f} | {d['exact_matches']}/{d['n_runs']} |")
    lines += ["", "Speedup is the median of ten per-question ratios. Within a question, each condition's throughput is the median of its two repeats. The headline is throughput, not necessarily time to complete the same answer: some English outputs differ in length or content.", "",
              "| K | en pooled tau | hi pooled tau | ur pooled tau | ur < hi questions | ur < en questions | hi < en questions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in KS:
        d = final["paired_tau"][str(k)]
        lines.append(f"| {k} | " + " | ".join(f"{final['langs'][l]['by_k'][str(k)]['tau_pooled']:.3f}" for l in LANGS) + f" | {d['ur_vs_hi']['lower_count']}/10 | {d['ur_vs_en']['lower_count']}/10 | {d['hi_vs_en']['lower_count']}/10 |")
    lines += ["", "Tau includes the correction/bonus token from the target. It is not an acceptance percentage, and tau ≈ 2 does not mean the draft can never produce more than two correct tokens. Increasing K from 6 to 8 gave Urdu little additional pooled throughput per target pass in this sample.", "",
              "## Sensitivity checks", "", "| Language | K=2 repeat 0 | K=2 repeat 1 | K=2 matching outputs only | Matching subset questions | K=2 full 128-token outputs only | Full-length questions |", "|---|---:|---:|---:|---:|---:|---:|"]
    for l in LANGS:
        d = final["langs"][l]["by_k"]["2"]; m=d["matching_only"]; f=d["full_length_only"]
        lines.append(f"| {l} | {d['repeat_speedups'][0]:.3f} | {d['repeat_speedups'][1]:.3f} | {m['median']:.3f} | {m['n_prompts']} | {f['median']:.3f} | {f['n_prompts']} |")
    lines += ["", "Matching-only and full-length analyses select different subsets. They test sensitivity and must not replace the prespecified all-run results. Matching-only runs are not a new independent replication.", "",
              "K=2 paired-question bootstrap intervals for the median speedup (descriptive resampling of this question set, not a population guarantee): " + "; ".join(
                  f"{l} {final['langs'][l]['by_k']['2']['bootstrap_median_interval']}" for l in LANGS) + ".",
              "One Hindi question has a K=2 prompt speedup above 1; the claim concerns the language median, not every individual question.", "",
              "| K | Matching-only ur < hi | Common questions | Matching-only ur < en | Common questions |", "|---|---:|---:|---:|---:|"]
    for k in KS:
        d=final["matching_only_paired_tau"][str(k)]
        lines.append(f"| {k} | {d['ur_vs_hi']['lower_count']} | {d['ur_vs_hi']['n_pairs']} | {d['ur_vs_en']['lower_count']} | {d['ur_vs_en']['n_pairs']} |")
    lines += ["", "Both penalty settings show lower pooled Urdu tau, but the earlier run had more clock variation and repetition. Changing repetition penalty changes the token trajectory; the comparison cannot isolate repetition as the causal explanation.", "",
              "## Timing explanation: measured versus inferred", "",
              "- Calibration: target 34.64 ms/generated token; draft 23.13 ms/generated token; c = 0.668. Each measurement is one 64-token generation after warm-up, includes prefill and generation bookkeeping, and is not an isolated kernel measurement.",
              "- Both models were resident on the GPU. 'Each model alone' means generating separately, not loading only one model into memory.",
              "- `speedup ≈ tau / (1 + K*c)` assumes K draft steps per round and verification costs comparable to a baseline target step. Real workloads include EOS/end-budget truncation, cache work, verification length and bookkeeping.",
              f"- Approximation error against the final medians reaches {100*final['prediction_abs_error_max']:.1f}% relative. Agreement is useful, but does not prove the mechanism or exclude implementation bugs.",
              "- Python/launch overhead, memory bandwidth, cache work and verification cost are possible contributors. No profiler trace separates them. Do not promise batching or another engine will fix the result without testing it.",
              "- Clock readings were collected after each run, not continuously. They detect some changes but do not establish steady clock, temperature or no memory spill throughout a run.", "",
              "## Output correctness and language limitations", "",
              f"- Final run: {final['exact_matches']}/{final['n_spec']} speculative runs exactly matched ({100*final['exact_matches']/final['n_spec']:.1f}%). Baseline repeats matched. The cause of all historical mismatches has not been established.",
              "- FP16 accumulation differences are plausible, but the saved output file contains only canonical baseline answers, not speculative IDs, first-divergence logits or precision-controlled reruns. Non-monotonic mismatch versus K does not rule out a bug.",
              "- The previous FP16 spacing argument was numerically wrong: around 31.75, binary16 spacing is 2^(4-10) = 0.015625. A 0.015625 gap is one ULP, not half a ULP. The reported smoke-test logits are not stored here for independent verification.",
              "- The historical script scores come from canonical baseline text and were copied to all conditions. They cannot certify the language of a divergent speculative answer. Arabic script is not proof of Urdu, and none of the script checks assess fluency or factuality.",
              "- English has some early-EOS answers; all Hindi/Urdu baseline runs reach 128 tokens. Equal token ceilings do not imply equal semantic content or completed answers.",
              "- Tokens/word uses whitespace segments, not a language-aware word counter. Characters include whitespace/punctuation and Unicode code points, not comparable units of meaning. The observed tokenization burden is descriptive, not a tokenizer-only causal experiment.",
              "- Ten hand-selected questions limit generalization. Hindi/English similarity is not a formal equivalence test; Urdu differences cannot identify drafter weakness separately from target behavior, task, tokenization or translation quality.", "",
              "- K=1 was not measured in either historical sweep. 'No tested K wins' is supported; 'no possible K can win' is not. Future defaults include K=1, since shorter speculation can reduce wasted draft work.", "",
              "## Uncertainty", "",
              "- Two-repeat medians equal two-value means; a slow trial still moves them. More repeats would improve timing precision, but cannot enlarge the question sample.",
              "- `speedup_sd` in historic summaries describes variation across questions, not uncertainty in the median or GPU timing stability. Chart labels now state this explicitly.",
              "- `analysis/audit.json` includes paired question bootstrap intervals (10,000 resamples, seed 0), repeat-specific estimates and descriptive sign tests. Curated prompts are not a random sample of a language population; repeated K comparisons reuse the same questions.", "",
              "## Correcting the paper comparison", "",
              "The pasted advice that Leviathan et al. used large batches is incorrect. Their empirical walltime experiment used batch size 1 on one TPU-v4, with T5-XXL and much smaller T5 drafters. Our model pair, task, hardware and implementation differ. Say those differences matter; do not claim batch size 1 inherently prevents a win. [Original paper, §4.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).", "",
              "## Slide wording and Q&A", "",
              f"> In our Qwen/Hugging Face laptop benchmark, speculation reduced target forward calls but every tested K lost median throughput. Urdu produced fewer tokens per target pass than Hindi across all ten questions; Hindi tracked English on that metric. Exact output matches were {100*final['exact_matches']/final['n_spec']:.1f}%, so practical numerical equivalence remains a limitation.", "",
              "- Why slower? Drafting and verification added time that outweighed the saved target calls. Our approximate cost model is consistent with this; it does not isolate the hardware/framework cause.",
              "- Does it work beyond English? This benchmark cannot answer that universally. The Hindi acceptance result already contradicts a blanket non-English penalty.",
              "- Does a better GPU fix Urdu? Unknown. Higher acceptance needs better target/draft agreement; speed also depends on costs. No cross-hardware data is saved here.",
              "- Is output identical? The algorithm has an equivalence guarantee under its assumptions; this FP16 implementation did not match every recorded baseline.",
              "- What would change the conclusions? An optimized, instrumented rerun could change speed; broader tasks/models could change the language ordering; saved divergences and precision controls could resolve correctness.", "",
              "## Ready for slides", "",
              "Use the final local tables/chart with the stated scope and correctness limitation. Further GPU experiments are needed for stronger claims about T4 replication, overhead causality, or correctness of every historical mismatch. The department template itself and whether its sample-paper list is mandatory are not present in this folder.", ""]
    probe_path = HERE / "analysis" / "correctness_probe.json"
    if probe_path.exists():
        p = json.loads(probe_path.read_text(encoding="utf-8"))
        probe_lines = ["## Targeted correctness probe (additional evidence)", "",
                       "This is an untimed diagnostic, not part of the 600-trial speed sweep. It uses cached model revisions and the same FP16/model/penalty settings. Saved IDs/text/settings are in `analysis/correctness_probe.json`; selected raw/processed logits are in `analysis/correctness_probe_logits.pt`.", "",
                       f"- Question: {p['prompt']} (en question 5, K=4). Baseline repeat matches: {p['baseline_repeat_matches']}; speculative repeat matches: {p['speculative_repeat_matches']}; speculative/baseline match: {p['matches_baseline']}."]
        if p.get("scores"):
            s = p["scores"]
            probe_lines += [f"- First divergence: generated token {p['first_divergence_token_0based'] + 1} (one-based). The baseline selects `exams`; speculation selects `exam`. The score gap between these candidates is {s['baseline_margin_a_minus_b']:.8f} in baseline scoring and {s['speculative_margin_a_minus_b']:.8f} in speculative scoring.",
                            f"- Reapplying the repetition processor reconstructs both selected IDs: {s['argmax_reconstruction_matches']}. This directly supports a near-tie explanation for this observed divergence; it disproves the old 'always late' claim."]
        if p.get("same_cache_shape_control"):
            c = p["same_cache_shape_control"]
            probe_lines += [f"- Control: target-only forwards from cloned, freshly rebuilt identical KV caches, scoring one versus several positions. Argmax changed: {c['rank_changed']}; maximum raw-score difference {c['max_abs_raw_difference']:.8f}. This control removes draft generation and speculative cache rollback from the comparison.",
                            "- Scope: the probe demonstrates numerical sensitivity for one prefix. It does not certify semantic equivalence of complete answers or diagnose every historical mismatch."]
        # Put this next to the historical correctness limitations, before uncertainty.
        insert = lines.index("## Uncertainty")
        lines[insert:insert] = probe_lines + [""]
    (HERE / "DATA_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    audits, table = {}, []
    for name in ("results", "results_rp105"):
        a, t = audit(HERE / name); audits[name] = a; table += t
    out = HERE / "analysis"; out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    with (out / "prompt_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0]); w.writeheader(); w.writerows(table)
    write_review(audits)
    for name, a in audits.items():
        print(name, "rows", a["n_runs"], "errors", a["errors"], "exact", a["exact_matches"], "/", a["n_spec"])
        print("paired tau", json.dumps(a["paired_tau"]))
    if any(a["errors"] for a in audits.values()):
        raise SystemExit("Data integrity errors: inspect analysis/audit.json")


if __name__ == "__main__":
    main()
