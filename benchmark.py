"""
Speculative decoding benchmark: English vs Hindi vs Urdu on a consumer GPU.

Draft : Qwen/Qwen2.5-0.5B-Instruct
Target: Qwen/Qwen2.5-3B-Instruct    (both fp16, greedy)

What it measures
----------------
  c        cost of one draft forward pass / one target forward pass (batch 1)
  tau      generated tokens / target forward passes (normal decoding = 1.0)
  speedup  speculative tok/s / normal tok/s
           theory:  speedup ~= tau / (1 + K*c)

Why the draft strategy is re-pinned before EVERY generate() call
---------------------------------------------------------------
The transformers 4.57 defaults are NOT a dynamic draft-length heuristic:
  num_assistant_tokens           = 20
  num_assistant_tokens_schedule  = "constant"
  assistant_confidence_threshold = 0.4
So K is a constant 20, cut short early by a confidence stopping criterion.
That threshold is not static either: when scikit-learn is installed,
AssistedCandidateGenerator.update_candidate_strategy fits an ROC curve and
writes the new threshold back onto draft.generation_config, where it SURVIVES
into the next generate() call. Left alone, the effective draft length drifts
across prompts, so language gets confounded with run order.

set_draft_strategy() rewrites all three before every call and pins the
threshold to 0, which disables the early exit (gated on `> 0` in
_get_stopping_criteria) and the ROC write-back (gated on truthiness).

Other methodology notes
-----------------------
  * repetition_penalty is forced to 1.0. Qwen ships 1.05 (target) and 1.1
    (draft) in generation_config, so "greedy" would otherwise not be argmax
    and the two models would be penalised differently.
  * Condition order is shuffled with a fixed seed, so thermal drift is spread
    across conditions instead of always landing on whatever ran last.
  * Every condition is repeated --repeats times and aggregated by MEDIAN, so a
    single transient slowdown cannot move a language's number.
  * tau is reported pooled (total tokens / total passes) as well as averaged.

Usage:
  python benchmark.py --limit 2 --repeats 1          # quick smoke test
  python benchmark.py                                # full K sweep
  python benchmark.py --k 4                          # one K only
  python benchmark.py --target Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import csv
import json
import random
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
LANG_NAMES = {"en": "English", "hi": "Hindi", "ur": "Urdu"}
SYSTEM = "Answer in the same language as the question."

# Unicode ranges used to check the answer is really in the asked language.
SCRIPTS = {
    "en": [(0x0041, 0x005A), (0x0061, 0x007A)],                      # Latin
    "hi": [(0x0900, 0x097F)],                                        # Devanagari
    "ur": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],  # Arabic script
}
LANG_OK_THRESHOLD = 0.7


def script_ratio(text, lang):
    """Share of letters in the output that belong to the expected script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hit = sum(any(a <= ord(c) <= b for a, b in SCRIPTS[lang]) for c in letters)
    return hit / len(letters)


class PassCounter:
    """Counts forward passes of a model (one per decoding step / verification)."""

    def __init__(self, model):
        self.n = 0
        model.register_forward_hook(self._hook)

    def _hook(self, *_):
        self.n += 1


def load(name):
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda")
    model.eval()
    return model


def set_draft_strategy(draft, k):
    """Pin the speculation strategy. Must run before EVERY assisted generate():
    transformers mutates assistant_confidence_threshold in place and the change
    persists on draft.generation_config across calls."""
    gc = draft.generation_config
    gc.num_assistant_tokens = int(k)
    gc.num_assistant_tokens_schedule = "constant"
    gc.assistant_confidence_threshold = 0.0   # disables early exit + ROC write-back


def encode(tok, prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to("cuda")


GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
              repetition_penalty=1.0)   # Qwen ships 1.05/1.1; force true argmax


@torch.inference_mode()
def generate(target, tok, inputs, max_new, counter, draft=None, k=None):
    kwargs = dict(max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **GREEDY)
    if draft is not None:
        set_draft_strategy(draft, k)
        kwargs["assistant_model"] = draft
    counter.n = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = target.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    return new_ids.tolist(), dt, counter.n


@torch.inference_mode()
def calibrate_pass_cost(target, draft, tok, counter, n=64):
    """Seconds per forward pass for each model alone, batch 1, with a KV cache.
    Plain greedy decoding is exactly one forward pass per token, so sec/token is
    sec/pass. Returns (target_sec_per_pass, draft_sec_per_pass, c)."""
    inputs = encode(tok, "Explain how rain forms.")
    out = {}
    for name, model in (("target", target), ("draft", draft)):
        for _ in range(2):   # warm up this model's kernels
            model.generate(**inputs, max_new_tokens=8, min_new_tokens=8,
                           pad_token_id=tok.eos_token_id, **GREEDY)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=n, min_new_tokens=n,
                       pad_token_id=tok.eos_token_id, **GREEDY)
        torch.cuda.synchronize()
        out[name] = (time.perf_counter() - t0) / n
    counter.n = 0
    return out["target"], out["draft"], out["draft"] / out["target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "ur"])
    ap.add_argument("--limit", type=int, default=None, help="prompts per language (default: all)")
    ap.add_argument("--k", nargs="+", type=int, default=[2, 4, 6, 8],
                    help="draft lengths to sweep (HF unpinned behaves like K=20)")
    ap.add_argument("--repeats", type=int, default=2, help="trials per condition; median reported")
    ap.add_argument("--seed", type=int, default=0, help="seed for condition shuffling")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available - check the venv's torch build."
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    tok = AutoTokenizer.from_pretrained(args.target)
    print(f"Loading target {args.target} ...")
    target = load(args.target)
    print(f"Loading draft  {args.draft} ...")
    draft = load(args.draft)

    used = torch.cuda.memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"VRAM after loading: {used:.1f} / {total:.1f} GB")
    if used > 0.85 * total:
        print("WARNING: VRAM nearly full. On Windows the driver silently spills into shared RAM and"
              " timings become meaningless. Re-run with --target Qwen/Qwen2.5-1.5B-Instruct.")

    counter = PassCounter(target)
    prompts = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))

    # Warm-up (CUDA kernels, caches) - not measured.
    warm = encode(tok, "Hello, how are you?")
    generate(target, tok, warm, 32, counter)
    generate(target, tok, warm, 32, counter, draft, k=args.k[0])

    print("\nCalibrating forward-pass cost (batch 1) ...")
    t_pass, d_pass, c = calibrate_pass_cost(target, draft, tok, counter)
    k0 = args.k[0]
    print(f"  target {t_pass * 1e3:.1f} ms/pass | draft {d_pass * 1e3:.1f} ms/pass | c = {c:.2f}")
    print(f"  a win needs tau > 1 + K*c, i.e. tau > {1 + k0 * c:.1f} at K={k0}")

    # Build the full condition list, then shuffle per repeat so thermal drift is
    # spread across conditions instead of always hitting whatever runs last.
    conditions = []
    for lang in args.langs:
        for i in range(len(prompts[lang][: args.limit])):
            conditions.append((lang, i, 0))                 # k=0 -> baseline
            for k in args.k:
                conditions.append((lang, i, k))

    out_dir = HERE / args.out
    out_dir.mkdir(exist_ok=True)
    rows, canonical, ids_by_row = [], {}, []
    rng = random.Random(args.seed)
    total_runs = len(conditions) * args.repeats
    done = 0
    t_start = time.perf_counter()

    for rep in range(args.repeats):
        order = conditions[:]
        rng.shuffle(order)
        for lang, i, k in order:
            inputs = encode(tok, prompts[lang][i])
            ids, dt, passes = generate(target, tok, inputs, args.max_new, counter,
                                       draft=(draft if k else None), k=(k or None))
            key = (lang, i)
            if k == 0 and key not in canonical:
                canonical[key] = ids
            done += 1
            tau = len(ids) / max(passes, 1)
            ids_by_row.append(ids)
            rows.append(dict(
                repeat=rep, lang=lang, prompt_id=i, k=k,
                mode=("baseline" if k == 0 else f"spec_k{k}"),
                tokens=len(ids), sec=round(dt, 3), tps=round(len(ids) / dt, 2),
                passes=passes, tau=round(tau, 3),
                matches_baseline=None,   # resolved after the loop, see below
                wall=round(time.perf_counter() - t_start, 1),
            ))
            label = "baseline" if k == 0 else f"K={k}"
            print(f"  [{done:3d}/{total_runs}] {lang} p{i} {label:>9s}"
                  f" | {len(ids) / dt:6.1f} tok/s | {passes:4d} passes | tau {tau:.2f}")

    # Resolve identical-output checks only now. The run order is shuffled, so a
    # speculative run can happen before that prompt's first baseline; comparing
    # inside the loop would score those as "not identical".
    for r, ids in zip(rows, ids_by_row):
        r["matches_baseline"] = ids == canonical[(r["lang"], r["prompt_id"])]

    # Texts + language check, from the canonical baseline output of each prompt.
    samples, meta = [], {}
    for (lang, i), ids in canonical.items():
        text = tok.decode(ids, skip_special_tokens=True)
        sr = script_ratio(text, lang)
        meta[(lang, i)] = dict(
            lang_ok=sr >= LANG_OK_THRESHOLD,
            script_ratio=round(sr, 2),
            tokens_per_word=round(len(ids) / max(len(text.split()), 1), 2),
            chars_per_token=round(len(text) / max(len(ids), 1), 2),
        )
        samples.append(dict(lang=lang, prompt_id=i, prompt=prompts[lang][i],
                            output=text, **meta[(lang, i)]))
    samples.sort(key=lambda s: (args.langs.index(s["lang"]), s["prompt_id"]))

    for r in rows:
        r.update(meta[(r["lang"], r["prompt_id"])])

    with open(out_dir / "raw.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    (out_dir / "outputs.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarise(rows, args, gpu, t_pass, d_pass, c)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(summary, args)
    plot(summary, args, out_dir)
    print(f"\nSaved: {out_dir}")


def med(xs):
    return round(st.median(xs), 3)


def summarise(rows, args, gpu, t_pass, d_pass, c):
    """Median over repeats per (lang, prompt, k), then median over prompts."""
    ok = [r for r in rows if r["lang_ok"]]
    summary = dict(
        gpu=gpu, target=args.target, draft=args.draft, max_new=args.max_new,
        repeats=args.repeats, seed=args.seed, k_sweep=args.k,
        target_ms_per_pass=round(t_pass * 1e3, 2),
        draft_ms_per_pass=round(d_pass * 1e3, 2),
        c=round(c, 3),
        langs={},
    )
    for lang in args.langs:
        lr = [r for r in ok if r["lang"] == lang]
        all_lang = [r for r in rows if r["lang"] == lang]
        if not lr or not all_lang:
            continue
        pids = sorted({r["prompt_id"] for r in lr})
        base = {}
        for p in pids:
            b = [r["tps"] for r in lr if r["prompt_id"] == p and r["k"] == 0]
            if b:
                base[p] = st.median(b)
        if not base:
            continue
        drift = [r["tps"] for r in all_lang if r["k"] == 0]
        entry = dict(
            n_prompts=len(base),
            lang_ok_rate=round(len({r["prompt_id"] for r in lr})
                               / len({r["prompt_id"] for r in all_lang}), 2),
            base_tps=med(list(base.values())),
            baseline_tps_min=round(min(drift), 2),
            baseline_tps_max=round(max(drift), 2),
            tokens_per_word=med([r["tokens_per_word"] for r in lr if r["k"] == 0]),
            chars_per_token=med([r["chars_per_token"] for r in lr if r["k"] == 0]),
            by_k={},
        )
        for k in args.k:
            kr = [r for r in lr if r["k"] == k]
            if not kr:
                continue
            speedups, taus = [], []
            for p in base:
                pr = [r for r in kr if r["prompt_id"] == p]
                if not pr:
                    continue
                speedups.append(st.median([r["tps"] for r in pr]) / base[p])
                taus.append(st.median([r["tau"] for r in pr]))
            if not speedups:
                continue
            tau_pooled = sum(r["tokens"] for r in kr) / max(sum(r["passes"] for r in kr), 1)
            entry["by_k"][str(k)] = dict(
                spec_tps=med([r["tps"] for r in kr]),
                speedup=med(speedups),
                speedup_mean=round(st.mean(speedups), 3),
                speedup_sd=round(st.pstdev(speedups), 3) if len(speedups) > 1 else 0.0,
                tau=med(taus),
                tau_pooled=round(tau_pooled, 3),
                predicted_speedup=round(tau_pooled / (1 + k * c), 3),
                identical_rate=round(sum(bool(r["matches_baseline"]) for r in kr) / len(kr), 2),
            )
        if entry["by_k"]:
            best = max(entry["by_k"], key=lambda k: entry["by_k"][k]["speedup"])
            entry["best_k"] = int(best)
            entry["best_speedup"] = entry["by_k"][best]["speedup"]
        summary["langs"][lang] = entry
    return summary


def report(summary, args):
    print("\n=== Forward-pass cost (batch 1) ===")
    print(f"target {summary['target_ms_per_pass']:.1f} ms/pass | "
          f"draft {summary['draft_ms_per_pass']:.1f} ms/pass | c = {summary['c']:.2f}")
    print(f"\n=== Per language (median over prompts, {summary['repeats']} repeats each) ===")
    for lang, s in summary["langs"].items():
        print(f"\n{LANG_NAMES[lang]}  (n={s['n_prompts']} prompts,"
              f" right language {s['lang_ok_rate']:.0%})")
        print(f"  normal {s['base_tps']:.1f} tok/s"
              f"  (baseline spread {s['baseline_tps_min']:.1f}-{s['baseline_tps_max']:.1f} tok/s)")
        print(f"  tokens/word {s['tokens_per_word']:.2f} | chars/token {s['chars_per_token']:.2f}")
        print(f"  {'K':>3s} {'spec tok/s':>11s} {'speedup':>8s} {'tau':>6s} {'tau pooled':>11s}"
              f" {'predicted':>10s} {'identical':>10s}")
        for k, d in s["by_k"].items():
            print(f"  {k:>3s} {d['spec_tps']:>11.1f} {d['speedup']:>7.2f}x {d['tau']:>6.2f}"
                  f" {d['tau_pooled']:>11.2f} {d['predicted_speedup']:>9.2f}x"
                  f" {d['identical_rate']:>9.0%}")
        if "best_k" in s:
            tail = "" if s["best_speedup"] > 1 else "   (still slower than normal decoding)"
            print(f"  best: K={s['best_k']} at {s['best_speedup']:.2f}x{tail}")
    wins = [s["best_speedup"] for s in summary["langs"].values() if "best_speedup" in s]
    if wins and max(wins) <= 1:
        print(f"\nNo K beat normal decoding (best {max(wins):.2f}x). With c = {summary['c']:.2f},"
              f" a win needs tau > 1 + K*c, and the measured tau never gets there."
              f"\nThat is the honest finding: on this GPU + HF, per-call overhead dominates."
              f" The language effect on tau is a separate, independent result.")


def plot(summary, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    langs = [l for l in args.langs if l in summary["langs"]]
    if not langs:
        return
    colors = {"en": "#4C6EF5", "hi": "#F59F00", "ur": "#E8590C"}

    # --- main slide chart: speedup vs K, and tau vs K -----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    for l in langs:
        by_k = summary["langs"][l]["by_k"]
        ks = sorted(int(k) for k in by_k)
        if not ks:
            continue
        ax1.plot(ks, [by_k[str(k)]["speedup"] for k in ks], "o-",
                 color=colors.get(l), label=LANG_NAMES[l])
        ax2.plot(ks, [by_k[str(k)]["tau_pooled"] for k in ks], "o-",
                 color=colors.get(l), label=LANG_NAMES[l])
    ax1.axhline(1.0, color="#868E96", lw=1, ls="--")
    ax1.text(0.02, 1.01, "no speedup", transform=ax1.get_yaxis_transform(),
             fontsize=8, color="#868E96", va="bottom")
    ax1.set_xlabel("K (draft tokens per round)")
    ax1.set_ylabel("Speedup vs normal decoding")
    ax1.set_title("Speedup", fontsize=10)
    ax2.set_xlabel("K (draft tokens per round)")
    ax2.set_ylabel("tau (tokens per target pass)")
    ax2.set_title("Draft acceptance", fontsize=10)
    for ax in (ax1, ax2):
        ax.legend(fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"{args.draft.split('/')[-1]} -> {args.target.split('/')[-1]}"
                 f" on {summary['gpu']}   (c = {summary['c']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png")
    plt.close(fig)

    # --- best-K bar chart, for the README -----------------------------------
    have = [l for l in langs if "best_k" in summary["langs"][l]]
    if not have:
        return
    vals = [summary["langs"][l]["best_speedup"] for l in have]
    bks = [summary["langs"][l]["best_k"] for l in have]
    sds = [summary["langs"][l]["by_k"][str(b)]["speedup_sd"] for l, b in zip(have, bks)]
    taus = [summary["langs"][l]["by_k"][str(b)]["tau_pooled"] for l, b in zip(have, bks)]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    bars = ax.bar([LANG_NAMES[l] for l in have], vals, yerr=sds, capsize=4,
                  color=[colors.get(l, "#4C6EF5") for l in have], edgecolor="none", width=0.55)
    ax.axhline(1.0, color="#868E96", lw=1, ls="--")
    ax.text(len(have) - 0.5, 1.0, " no speedup", va="bottom", ha="right",
            fontsize=8, color="#868E96")
    for b, v, t, bk in zip(bars, vals, taus, bks):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                f"{v:.2f}x\nK={bk}, tau {t:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Speedup vs normal decoding (best K)")
    ax.set_ylim(0, max(max(vals) * 1.45 + 0.2, 1.15))
    ax.set_title(f"Speculative decoding by language, best K per language\n"
                 f"{args.draft.split('/')[-1]} -> {args.target.split('/')[-1]}, {summary['gpu']}",
                 fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
