"""Synchronize notebook analysis with benchmark.py; preserve executed evidence.

Local notebook imports the shared functions. Kaggle notebooks embed copies of
the same aggregation/evidence functions so they remain upload-and-run files.
All changed code cells are unexecuted; historical outputs are archived once.
"""
import ast
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "benchmark.py").read_text(encoding="utf-8")
tree = ast.parse(source)
functions = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def set_source(cell, text):
    if "".join(cell["source"]) != text:
        cell["source"] = text.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def refresh(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    local = path.name == "benchmark.ipynb"
    seven = "7b" in path.name
    if local:
        archived = HERE / "results_rp105" / "benchmark_executed.ipynb"
        if not archived.exists():
            shutil.copyfile(path, archived)
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            text = text.replace("Output identical", "Output equivalence is theoretical; measured matches are recorded")
            text = text.replace("output identical", "output equivalence is theoretical; measured matches are recorded")
            text = text.replace("identical outputs", "outputs checked for exact matches")
            text = text.replace("one bad trial cannot move it", "two-repeat medians are still sensitive to a bad trial")
            text = text.replace("one bad trial cannot move", "two-repeat medians are still sensitive to")
            text = text.replace("right-language", "script-screened")
            text = text.replace("really in the asked language", "in the expected script (not a quality check)")
            if "## 9. How much" in text or "## 8. `c`" in text:
                text = "## Calibration: generation-cost proxy\n\nMeasure synchronized wall time per generated token, including prefill and bookkeeping, with both models resident. Use three interleaved repeats. This is a cost proxy; a profiler is needed to identify overhead sources. `tau / (1 + K*c)` is an approximation, not a guaranteed bound for the measured implementation.\n"
            if "## What to take back" in text:
                text = "## What to take back to the slides\n\nReport measured throughput and tokens per target pass with the model pair, device placement and prompt count. Save all token IDs and match rates. A matching cost estimate does not prove an overhead cause. These notebooks are proposed checks; no Kaggle results are saved in the local project.\n"
            if "## 13. Summary" in text or "## 11. Results" in text:
                text = "## Summary\n\nScreen every actual output by Unicode script; this does not assess language quality. Use median throughput within a prompt and then median prompt speedup ratios. Tau includes the target bonus/correction token. With two repeats the median equals the mean. SD across prompt ratios is variation across questions, not a confidence interval.\n"
            if "## 12. Language check" in text:
                text = "## Output evidence and script screen\n\nKeep all baseline and speculative IDs/text, exact match flags and first divergent token positions. Screen every output independently. Whitespace segments and Unicode code points are descriptive text measures, not comparable units of meaning.\n"
            if "## 2. Settings" in text and local:
                text = "## 2. Settings\n\nThe measured historical sweep used K=2,4,6,8 and two repeats. New defaults also test K=1 and use three repeats. Set a new OUTNAME for each run. A fixed shuffle seed reproduces order, not hardware timing. Start the notebook in this project folder, or explicitly set HERE to its path.\n"
            if "## 3. Language check" in text or (not local and "## 3. Language" in text):
                text = "## 3. Script screen\n\nCount the fraction of alphabetic Unicode code points in the expected script. A 70% threshold screens script switching; it does not establish fluency, factuality or even language identity. New code checks each actual output, including divergent speculative answers.\n"
            if "## 6. Pinning" in text:
                text = "## 6. Pinning the draft strategy\n\nIn Transformers 4.57.6, default K=20 is constant with a confidence gate. With scikit-learn, confidence state can be written back across calls. Before every assisted call, pin K, the constant schedule and confidence threshold 0, disabling that adaptive behavior. K remains a maximum: EOS and the final generation budget may shorten a round.\n"
            if "## 7. The two core" in text or "## 7. Core functions" in text:
                text = "## 7. Core functions\n\nEncode the question with Qwen's chat template. Time synchronized target.generate, with assistant_model=draft for speculation. Set repetition_penalty explicitly; HF reuses target settings/processors for the assistant. do_sample=False selects the largest processed score, not necessarily the largest raw model logit.\n"
            if "## 11. Main sweep" in text or "## 10. The sweep" in text:
                text = "## Shuffled sweep\n\nBuild baseline and speculative conditions for every question, shuffle each repeat, and record time, tokens and target calls. Shuffling spreads drift; it does not eliminate it. Medians of two trials equal their mean. Exact-match checks run after baselines are available. Saved IDs/text allow later inspection of divergences.\n"
            if "## 14. Charts" in text:
                text = "## Charts\n\nShow measured median speedup and pooled tokens per target pass. The latter includes correction/bonus tokens. Error bars on the best-K bars show SD across question ratios, not a confidence interval. Compare only tested K values.\n"
            if cell is notebook["cells"][0]:
                text = "# Speculative decoding: English, Hindi and Urdu\n\nCompare target-only and assisted generation with Qwen. Record throughput, target forward calls, exact output matches and script scores. Results describe the tested model pair and questions; saved numerical mismatches and quality limitations must be reported.\n"
            set_source(cell, text)
            continue
        if local and "TARGET  =" in text:
            text = text.replace('REPEATS = 2', 'REPEATS = 3').replace('OUTNAME = "results_rp105"', 'OUTNAME = "results_new"')
            text = text.replace('K_SWEEP = [2, 4, 6, 8]', 'K_SWEEP = [1, 2, 4, 6, 8]')
            text = text.replace('OUT.mkdir(exist_ok=True)', 'assert not (OUT / "raw.csv").exists(), "Choose a new OUTNAME; measurements already exist."\nOUT.mkdir(parents=True, exist_ok=True)\nimport sys\nsys.path.insert(0, str(HERE.resolve()))')
        elif local and text.startswith("def summarise(rows):"):
            text = '''from types import SimpleNamespace
from benchmark import summarise as aggregate, report

ARGS = SimpleNamespace(target=TARGET, draft=DRAFT, max_new=MAX_NEW,
                       repeats=REPEATS, seed=SEED, k=K_SWEEP,
                       rep_penalty=REP_PENALTY, langs=LANGS)
summary = aggregate(rows, ARGS, GPU, T_PASS, D_PASS, C)
report(summary, ARGS)
'''
        elif local and text.startswith("import matplotlib.pyplot"):
            text = '''from benchmark import plot
import matplotlib.pyplot as plt

plot(summary, ARGS, OUT)
for filename in ("sweep.png", "speedup.png"):
    plt.figure(figsize=(10, 4))
    plt.imshow(plt.imread(OUT / filename)); plt.axis("off"); plt.show()
'''
        elif local and "def pass_cost" in text:
            text = '''from benchmark import calibrate_pass_cost
CALIBRATION_SAMPLES = {}
T_PASS, D_PASS, C = calibrate_pass_cost(target, draft, tok, counter,
                                     rep_penalty=REP_PENALTY, samples=CALIBRATION_SAMPLES)
print(f"target {T_PASS*1e3:.2f} ms/token proxy; draft {D_PASS*1e3:.2f}; c={C:.3f}")
print("Includes prefill/bookkeeping. Overhead sources were not profiled.")
'''
        elif "samples, meta = [], {}" in text or text.startswith("meta, samples = {}, {}") or text.startswith("meta, samples = {}, []"):
            helpers = ("from benchmark import finalise_rows, text_metrics\n" if local else
                       "\n\n".join(functions[n] for n in ("text_metrics", "finalise_rows", "med", "summarise")) + "\n")
            prompts_var = "prompts" if local else "PROMPTS"
            text = helpers + f'''
evidence = finalise_rows(rows, ids_by_row, canonical, tok)
(OUT / "run_outputs.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
samples = []
for (lang, i), ids in canonical.items():
    answer, meta = text_metrics(tok, ids, lang)
    samples.append(dict(lang=lang, prompt_id=i, prompt={prompts_var}[lang][i], output=answer, **meta))
print("Exact speculative matches:", sum(r["matches_baseline"] for r in rows if r["k"]),
      "/", sum(bool(r["k"]) for r in rows))
'''
            if not local:
                text += '''
from types import SimpleNamespace
ARGS = SimpleNamespace(target=TARGET, draft=DRAFT, max_new=MAX_NEW,
                       repeats=REPEATS, seed=SEED, k=K_SWEEP,
                       rep_penalty=REP_PENALTY, langs=LANGS)
summary = summarise(rows, ARGS, GPU, T_PASS, D_PASS, C)
for lang, entry in summary["langs"].items():
    print(lang, entry["by_k"])
'''
        elif not local and "def pass_cost" in text:
            text = '''@torch.inference_mode()
def pass_costs(n=64):
    samples = {"target": [], "draft": []}
    models = (("target", target), ("draft", draft))
    def run(model, count):
        inputs = {k: v.to(model.device) for k, v in encode("Explain how rain forms.").items()}
        synchronize_all()
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=count, min_new_tokens=count,
                       pad_token_id=tok.eos_token_id, repetition_penalty=REP_PENALTY, **GREEDY)
        synchronize_all()
        return (time.perf_counter() - t0) / count
    for name, model in models:
        run(model, 8); run(model, 8)
    for rep in range(3):
        for name, model in (models if rep % 2 == 0 else models[::-1]):
            samples[name].append(run(model, n))
    return st.median(samples["target"]), st.median(samples["draft"]), samples

T_PASS, D_PASS, CALIBRATION_SAMPLES = pass_costs()
C = D_PASS / T_PASS
counter.n = 0
print(f"{TARGET}: {T_PASS*1e3:.2f} ms/token proxy; draft {D_PASS*1e3:.2f}; c={C:.3f}")
print("Includes prefill/bookkeeping; cross-device copies also matter in a two-GPU sweep.")
'''
        elif not local and "TARGET  =" in text:
            text = text.replace('K_SWEEP = [2, 4]', 'K_SWEEP = [1, 2, 4]')
            text = text.replace('OUT = Path("/kaggle/working/results"); OUT.mkdir(parents=True, exist_ok=True)',
                                'OUT = Path("/kaggle/working/results_new")\nassert not (OUT / "raw.csv").exists(), "Choose a new OUT folder."\nOUT.mkdir(parents=True, exist_ok=True)')
            text = text.replace('REPEATS = 1', 'REPEATS = 3')
        elif not local and "def generate(" in text:
            if not text.startswith('def synchronize_all():'):
                text = 'def synchronize_all():\n    for device in range(torch.cuda.device_count()):\n        torch.cuda.synchronize(device)\n\n' + text
            text = text.replace('torch.cuda.synchronize()', 'synchronize_all()')
        elif not local and seven and "N_GPU =" in text:
            text = text.replace('if N_GPU < 2:\n', 'assert N_GPU >= 2, "7B FP16 setup requires two T4 GPUs; use the 3B notebook for one GPU."\nif N_GPU < 2:\n')
        # Preserve precision in newly collected rows; historic files stay unchanged.
        text = text.replace('sec=round(dt, 3)', 'sec=dt').replace('tps=round(len(ids)/dt, 2)', 'tps=len(ids)/dt')
        text = text.replace('tau=round(tau, 3)', 'tau=tau')
        text = text.replace('ax2.set_title("Draft acceptance", fontsize=10)', 'ax2.set_title("Tokens per target pass (includes bonus)", fontsize=10)')
        text = text.replace('the two models are penalised by different amounts', 'the target settings are explicitly pinned')
        if '"raw.csv", "w"' in text and '\nmanifest = dict(' not in text:
            text += '''
import platform, transformers
manifest = dict(target=TARGET, draft=DRAFT, gpu=GPU, max_new=MAX_NEW,
                repeats=REPEATS, k_sweep=K_SWEEP, seed=SEED, rep_penalty=REP_PENALTY,
                python=platform.python_version(), torch=torch.__version__,
                transformers=transformers.__version__, cuda=torch.version.cuda,
                target_device=str(target.device), draft_device=str(draft.device),
                target_revision=getattr(target.config, "_commit_hash", None),
                draft_revision=getattr(draft.config, "_commit_hash", None),
                calibration_seconds_per_token=CALIBRATION_SAMPLES,
                calibration="three interleaved end-to-end measurements; includes prefill",
                target_generation_config=target.generation_config.to_dict(),
                draft_generation_config=draft.generation_config.to_dict())
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
'''
        set_source(cell, text)
    # All local code now represents the corrected workflow. Historical executed
    # outputs remain available in the archive rather than mixed with new source.
    if local:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []; cell["execution_count"] = None
    note = "\n\n**Data review:** see `DATA_REVIEW.md`. Historical final-run executions are archived in `results_rp105/benchmark_executed.ipynb`. This corrected notebook has not been run on the GPU.\n" if local else "\n\n**Proposed validation run:** no saved Kaggle executions are supplied with this project. The 7B notebook uses two GPUs and is not a controlled hardware-only comparison.\n"
    if note.strip() not in "".join(notebook["cells"][0]["source"]):
        notebook["cells"][0]["source"] += [note]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for name in ("benchmark.ipynb", "benchmark_kaggle.ipynb", "benchmark_kaggle_7b.ipynb"):
        refresh(HERE / name)
