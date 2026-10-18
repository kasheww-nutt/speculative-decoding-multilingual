"""Focused CPU checks of the changed benchmark and notebook workflow."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from audit_data import load_rows
from benchmark import finalise_rows, plot, summarise

HERE = Path(__file__).resolve().parent


def main():
    for name in ("benchmark.py", "audit_data.py", "refresh_notebooks.py", "probe_correctness.py"):
        ast.parse((HERE / name).read_text(encoding="utf-8"))
    for name in ("benchmark.ipynb", "benchmark_kaggle.ipynb", "benchmark_kaggle_7b.ipynb"):
        nb = json.loads((HERE / name).read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                # Compile notebook shell magics as pass; do not install anything.
                src = "".join(cell["source"])
                src = "\n".join("pass" if s.lstrip().startswith("!") else s for s in src.splitlines())
                compile(src, f"{name}:cell{i}", "exec")
                assert cell["execution_count"] is None and not cell["outputs"]
    # Independent saved-data audit already verifies summary arithmetic. Here
    # verify the production aggregator on real rows and documented precision.
    for name in ("results", "results_rp105"):
        saved = json.loads((HERE / name / "summary.json").read_text(encoding="utf-8"))
        args = SimpleNamespace(target=saved["target"], draft=saved["draft"], max_new=saved["max_new"],
                               repeats=saved["repeats"], seed=saved["seed"], k=saved["k_sweep"],
                               rep_penalty=saved.get("rep_penalty", 1.0), langs=["en", "hi", "ur"])
        rows = load_rows(HERE / name / "raw.csv")
        actual = summarise(rows, args, saved["gpu"], saved["target_ms_per_pass"] / 1000,
                           saved["draft_ms_per_pass"] / 1000, saved["c"])
        for lang in args.langs:
            for k in args.k:
                a, b = actual["langs"][lang]["by_k"][str(k)], saved["langs"][lang]["by_k"][str(k)]
                for metric in ("speedup", "tau", "tau_pooled", "identical_rate"):
                    assert abs(a[metric] - b[metric]) <= 0.00101, (name, lang, k, metric, a[metric], b[metric])
        if name == "results_rp105":
            # Draw derived charts; the historical result artifacts are unchanged.
            plot(saved, args, HERE / "analysis")
    # Regression: baseline-script metadata must never be copied over a divergent
    # speculative output. Record token substitutions and prefix length changes.
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "بارش" if ids == [1, 2] else "English"
    rows = [dict(lang="ur", prompt_id=0, repeat=0, k=0),
            dict(lang="ur", prompt_id=0, repeat=0, k=2),
            dict(lang="ur", prompt_id=0, repeat=1, k=2)]
    evidence = finalise_rows(rows, [[1, 2], [1, 3], [1, 2, 4]], {("ur", 0): [1, 2]}, Tokenizer())
    assert rows[0]["lang_ok"] and not rows[1]["lang_ok"]
    assert [r["matches_baseline"] for r in rows] == [True, False, False]
    assert [r["first_divergence_token_0based"] for r in evidence] == [None, 1, 2]
    assert evidence[1]["token_ids"] == [1, 3]
    probe_path = HERE / "analysis" / "correctness_probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        tensors = torch.load(HERE / "analysis" / "correctness_probe_logits.pt", map_location="cpu", weights_only=True)
        scores = probe["scores"]
        assert int(tensors["baseline_processed"].argmax()) == scores["baseline_selected_id"]
        assert int(tensors["speculative_processed"].argmax()) == scores["speculative_selected_id"]
        assert probe["baseline_repeat_matches"] and probe["speculative_repeat_matches"]
        assert scores["argmax_reconstruction_matches"] and probe["same_cache_shape_control"]["rank_changed"]
    print("PASS: source/notebook compilation, real-data aggregation, per-output screening and divergence evidence.")
    print("Derived charts saved to analysis/. No models or GPU experiments were run.")


if __name__ == "__main__":
    main()
