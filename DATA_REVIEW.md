# Data review before the presentation

Reviewed 12 October 2026. Reproduce the saved-data audit with `python audit_data.py`. No new GPU timing sweep was made; a separate targeted correctness probe is described below if present.

## Verdict

The saved local data supports a small, descriptive benchmark: no tested K improved median throughput; Urdu had lower tokens per target pass than Hindi on all ten translated questions. Hindi and English were similar on this metric. Exact output equivalence was not achieved on every run. The historical outputs alone cannot diagnose all mismatches; the targeted probe below supplies additional evidence for one case.

## Integrity and provenance

- `results`: 300 rows; 0 integrity/summary errors; 202/240 speculative runs exactly matched the baseline.
- `results_rp105`: 300 rows; 0 integrity/summary errors; 194/240 speculative runs exactly matched the baseline.
- Each dataset contains exactly 10 questions × 3 languages × 5 conditions × 2 repeats. Baseline passes equal generated tokens and both baseline repeats match their canonical baseline.
- Raw CSVs and historical summaries were preserved. SHA-256 fingerprints are in `analysis/audit.json`.
- Local notebook saved outputs support the final run. Both Kaggle notebooks have no saved executions; no T4 measurements or third hardware configuration are present. Count 600 recorded sweep trials, not 627 verified trials.
- Earlier `results/summary.json` omits repetition penalty. The 1.0 attribution comes from project notes, not a complete saved run manifest.

## Final run: what to put on the slides

| Language | Baseline tok/s | Best tested K | Median speedup | Pooled tokens/target pass at K=2 | Exact matches at K=2 |
|---|---:|---:|---:|---:|---:|
| en | 26.8 | 2 | 0.92× | 2.157 | 16/20 |
| hi | 27.1 | 2 | 0.85× | 2.159 | 18/20 |
| ur | 27.5 | 2 | 0.71× | 1.756 | 14/20 |

Speedup is the median of ten per-question ratios. Within a question, each condition's throughput is the median of its two repeats. The headline is throughput, not necessarily time to complete the same answer: some English outputs differ in length or content.

| K | en pooled tau | hi pooled tau | ur pooled tau | ur < hi questions | ur < en questions | hi < en questions |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 2.157 | 2.159 | 1.756 | 10/10 | 10/10 | 4/10 |
| 4 | 2.636 | 2.712 | 1.978 | 10/10 | 9/10 | 4/10 |
| 6 | 2.992 | 2.998 | 2.061 | 10/10 | 9/10 | 5/10 |
| 8 | 3.111 | 3.192 | 2.042 | 10/10 | 10/10 | 4/10 |

Tau includes the correction/bonus token from the target. It is not an acceptance percentage, and tau ≈ 2 does not mean the draft can never produce more than two correct tokens. Increasing K from 6 to 8 gave Urdu little additional pooled throughput per target pass in this sample.

## Sensitivity checks

| Language | K=2 repeat 0 | K=2 repeat 1 | K=2 matching outputs only | Matching subset questions | K=2 full 128-token outputs only | Full-length questions |
|---|---:|---:|---:|---:|---:|---:|
| en | 0.879 | 0.951 | 0.928 | 8 | 0.911 | 6 |
| hi | 0.830 | 0.874 | 0.853 | 9 | 0.853 | 10 |
| ur | 0.696 | 0.713 | 0.712 | 7 | 0.708 | 10 |

Matching-only and full-length analyses select different subsets. They test sensitivity and must not replace the prespecified all-run results. Matching-only runs are not a new independent replication.

K=2 paired-question bootstrap intervals for the median speedup (descriptive resampling of this question set, not a population guarantee): en [0.7674, 0.9632]; hi [0.8144, 0.8841]; ur [0.6526, 0.7519].
One Hindi question has a K=2 prompt speedup above 1; the claim concerns the language median, not every individual question.

| K | Matching-only ur < hi | Common questions | Matching-only ur < en | Common questions |
|---|---:|---:|---:|---:|
| 2 | 6 | 6 | 6 | 6 |
| 4 | 7 | 7 | 6 | 6 |
| 6 | 7 | 7 | 6 | 6 |
| 8 | 7 | 7 | 5 | 5 |

Both penalty settings show lower pooled Urdu tau, but the earlier run had more clock variation and repetition. Changing repetition penalty changes the token trajectory; the comparison cannot isolate repetition as the causal explanation.

## Timing explanation: measured versus inferred

- Calibration: target 34.64 ms/generated token; draft 23.13 ms/generated token; c = 0.668. Each measurement is one 64-token generation after warm-up, includes prefill and generation bookkeeping, and is not an isolated kernel measurement.
- Both models were resident on the GPU. 'Each model alone' means generating separately, not loading only one model into memory.
- `speedup ≈ tau / (1 + K*c)` assumes K draft steps per round and verification costs comparable to a baseline target step. Real workloads include EOS/end-budget truncation, cache work, verification length and bookkeeping.
- Approximation error against the final medians reaches 8.3% relative. Agreement is useful, but does not prove the mechanism or exclude implementation bugs.
- Python/launch overhead, memory bandwidth, cache work and verification cost are possible contributors. No profiler trace separates them. Do not promise batching or another engine will fix the result without testing it.
- Clock readings were collected after each run, not continuously. They detect some changes but do not establish steady clock, temperature or no memory spill throughout a run.

## Output correctness and language limitations

- Final run: 194/240 speculative runs exactly matched (80.8%). Baseline repeats matched. The cause of all historical mismatches has not been established.
- FP16 accumulation differences are plausible, but the saved output file contains only canonical baseline answers, not speculative IDs, first-divergence logits or precision-controlled reruns. Non-monotonic mismatch versus K does not rule out a bug.
- The previous FP16 spacing argument was numerically wrong: around 31.75, binary16 spacing is 2^(4-10) = 0.015625. A 0.015625 gap is one ULP, not half a ULP. The reported smoke-test logits are not stored here for independent verification.
- The historical script scores come from canonical baseline text and were copied to all conditions. They cannot certify the language of a divergent speculative answer. Arabic script is not proof of Urdu, and none of the script checks assess fluency or factuality.
- English has some early-EOS answers; all Hindi/Urdu baseline runs reach 128 tokens. Equal token ceilings do not imply equal semantic content or completed answers.
- Tokens/word uses whitespace segments, not a language-aware word counter. Characters include whitespace/punctuation and Unicode code points, not comparable units of meaning. The observed tokenization burden is descriptive, not a tokenizer-only causal experiment.
- Ten hand-selected questions limit generalization. Hindi/English similarity is not a formal equivalence test; Urdu differences cannot identify drafter weakness separately from target behavior, task, tokenization or translation quality.

- K=1 was not measured in either historical sweep. 'No tested K wins' is supported; 'no possible K can win' is not. Future defaults include K=1, since shorter speculation can reduce wasted draft work.

## Targeted correctness probe (additional evidence)

This is an untimed diagnostic, not part of the 600-trial speed sweep. It uses cached model revisions and the same FP16/model/penalty settings. Saved IDs/text/settings are in `analysis/correctness_probe.json`; selected raw/processed logits are in `analysis/correctness_probe_logits.pt`.

- Question: Give some tips for preparing for an exam. (en question 5, K=4). Baseline repeat matches: True; speculative repeat matches: True; speculative/baseline match: False.
- First divergence: generated token 13 (one-based). The baseline selects `exams`; speculation selects `exam`. The score gap between these candidates is 0.02975845 in baseline scoring and 0.00000000 in speculative scoring.
- Reapplying the repetition processor reconstructs both selected IDs: True. This directly supports a near-tie explanation for this observed divergence; it disproves the old 'always late' claim.
- Control: target-only forwards from cloned, freshly rebuilt identical KV caches, scoring one versus several positions. Argmax changed: True; maximum raw-score difference 0.03125000. This control removes draft generation and speculative cache rollback from the comparison.
- Scope: the probe demonstrates numerical sensitivity for one prefix. It does not certify semantic equivalence of complete answers or diagnose every historical mismatch.

## Uncertainty

- Two-repeat medians equal two-value means; a slow trial still moves them. More repeats would improve timing precision, but cannot enlarge the question sample.
- `speedup_sd` in historic summaries describes variation across questions, not uncertainty in the median or GPU timing stability. Chart labels now state this explicitly.
- `analysis/audit.json` includes paired question bootstrap intervals (10,000 resamples, seed 0), repeat-specific estimates and descriptive sign tests. Curated prompts are not a random sample of a language population; repeated K comparisons reuse the same questions.

## Correcting the paper comparison

The pasted advice that Leviathan et al. used large batches is incorrect. Their empirical walltime experiment used batch size 1 on one TPU-v4, with T5-XXL and much smaller T5 drafters. Our model pair, task, hardware and implementation differ. Say those differences matter; do not claim batch size 1 inherently prevents a win. [Original paper, §4.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

## Slide wording and Q&A

> In our Qwen/Hugging Face laptop benchmark, speculation reduced target forward calls but every tested K lost median throughput. Urdu produced fewer tokens per target pass than Hindi across all ten questions; Hindi tracked English on that metric. Exact output matches were 80.8%, so practical numerical equivalence remains a limitation.

- Why slower? Drafting and verification added time that outweighed the saved target calls. Our approximate cost model is consistent with this; it does not isolate the hardware/framework cause.
- Does it work beyond English? This benchmark cannot answer that universally. The Hindi acceptance result already contradicts a blanket non-English penalty.
- Does a better GPU fix Urdu? Unknown. Higher acceptance needs better target/draft agreement; speed also depends on costs. No cross-hardware data is saved here.
- Is output identical? The algorithm has an equivalence guarantee under its assumptions; this FP16 implementation did not match every recorded baseline.
- What would change the conclusions? An optimized, instrumented rerun could change speed; broader tasks/models could change the language ordering; saved divergences and precision controls could resolve correctness.

## Ready for slides

Use the final local tables/chart with the stated scope and correctness limitation. Further GPU experiments are needed for stronger claims about T4 replication, overhead causality, or correctness of every historical mismatch. The department template itself and whether its sample-paper list is mandatory are not present in this folder.
