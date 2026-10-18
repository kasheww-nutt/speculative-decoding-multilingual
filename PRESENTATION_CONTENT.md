# CA-3 presentation content

Prepared for the supplied six-page presentation template and NLP evaluation plan. This is slide content and speaker notes, not a completed PPTX.

## Scope and marking alignment

- Part A (CO4, 5 marks): advanced NLP model case study. Use the explanation of speculative decoding and the measured Qwen comparison.
- Part B (CO5, 7 marks): machine-translation case study. Use the published English-to-German study below. The local Hindi/Urdu questions are text generation, not translation.
- The evaluation plan does not specify whether the two parts require separate submissions or an original translation implementation. This draft covers both areas in one talk, with a published research case for Part B; it does not claim a local translation experiment.
- Keep the supplied order: title, timing, understanding, example, four-step explanation, wrap-up/questions.
- Count the timing page within the title's 45 seconds: 35 seconds on slide 1 plus 10 seconds on slide 2. Total planned speaking time: 300 seconds. Questions follow the five-minute talk.
- Fill in your actual name, section and PRN. These were not supplied.

## Slide 1 - Title (35 seconds)

### Text for the slide

**Topic Name**

Speculative Decoding

**Subtitle**

Machine Translation and Multilingual Inference

**One-line description**

A small model drafts text for a larger model to verify, aiming to reduce decoding time.

**Presenter line**

Presenter: [Your name] | Sec: [Your section] | PRN: [Your PRN]

### Speaker notes

When a language model generates an answer or a translation, it normally produces one token at a time. A token can be a word or part of a word, so a long answer requires many sequential model calls. My topic is speculative decoding: using a small model to draft several tokens for a larger model to check. I will explain the method, connect it to a published translation case, and evaluate it using my own laptop experiment.

## Slide 2 - How the 5 Minutes Break Down (10 seconds)

### Text for the slide

| Time | Section | Focus |
|---|---|---|
| ~45 sec | Title | Topic, motivation and roadmap |
| ~1 min | Understanding of the Topic | Definition, problem and NLP context |
| ~1.5 min | Example | Translation case and my measured comparison |
| ~1.5 min | Explanation of the Topic | Draft, verify, accept/correct, repeat |
| ~15 sec | Wrap-up / Q&A cue | Takeaway and questions |

### Speaker notes

I will first define the idea, then show the case study and measurements, explain the four steps, and finish with the main takeaway.

Do not read all the timing rows aloud. The 45-second title allowance includes this page.

## Slide 3 - Understanding of the Topic (60 seconds)

### Text for the slide

**Definition**

A small draft model proposes several tokens. A larger target model checks them together and keeps an agreed prefix.

**1. What problem does it solve?**

Normal decoding needs one sequential target-model call per new token.

**2. Where does it fit?**

Transformer inference, advanced NLP and neural machine translation.

**3. Why should the audience care?**

It can reduce translation and response latency when drafting is cheap and agreement is high.

### Speaker notes

There are two roles. The draft model is the fast proposer, and the target model is the model whose decoding behavior we want to retain. In ordinary decoding, the target runs again for each new token. In speculative decoding, the target can verify several proposed positions in one call. That reduces sequential target calls, although the draft still adds work. In translation, the source sentence stays as input and the models generate the target-language continuation. In a conversational model, the input is the prompt and the continuation is the answer. This is an inference technique: it changes how we generate, rather than requiring us to retrain the target. Whether it saves time depends on both agreement and execution cost.

## Slide 4 - Example (90 seconds)

### Text for the slide

**Scenario - left panel**

Published translation case:

- English to German: WMT EnDe
- Target: T5-XXL (11B)
- Draft: T5-small (77M)
- Goal: reduce translation decoding latency

My text-generation experiment:

- Qwen2.5 3B target + 0.5B draft
- RTX 4060 Laptop GPU, FP16
- 10 questions/language, 2 repeats
- 300 trials; up to 128 new tokens

**What Happens / Result - right panel**

Published MT result: **3.4x speedup**

Greedy decoding, K=7, batch 1, one TPU-v4.

My results at the best tested K=2:

| Language | Median speedup | Tokens/target call |
|---|---:|---:|
| English | 0.92x | 2.16 |
| Hindi | 0.85x | 2.16 |
| Urdu | 0.71x | 1.76 |

Urdu < Hindi on this metric in 10/10 questions at every tested K.

**Small source footer**

MT: Leviathan et al., ICML 2023, Table 2. Local: results_rp105. Different experiments, not a controlled hardware comparison.

### Speaker notes

The translation study uses English input and generates German output. It reports a 3.4-times speedup with a small T5 drafter, a much larger T5 target, and greedy decoding. These are published results, not measurements I reproduced.

I then tested the same general draft-and-verify idea on my laptop using Qwen models. I asked ten parallel questions in English, Hindi and Urdu and compared ordinary decoding against four draft lengths, with two shuffled repeats. The final run contained three hundred trials. Here, all tested draft lengths lost median throughput. The best was K equals two: English reached 0.92 times baseline speed, Hindi 0.85, and Urdu 0.71. Notice that Hindi and English produced almost the same number of tokens per target call, while Urdu produced fewer. The Urdu-versus-Hindi ordering held across all ten questions at every K. The published and local cases differ in task, models, hardware and implementation; this does not isolate a hardware cause.

### Layout note

Keep the two existing panels. Use the compact table rather than adding the full sweep chart to this crowded page. The full chart is available at analysis/sweep.png for discussion or a separate supporting handout.

## Slide 5 - Explanation of the Topic (90 seconds)

### Text for the slide

**Mechanism subtitle**

For translation, both models condition on the source sentence.

**1. Draft**

The small model proposes up to K output tokens.

**2. Verify**

The target scores the proposed positions together.

**3. Accept / correct**

Keep the agreed prefix; correct the first disagreement.

**4. Repeat**

Update the context and cache; continue until the stop condition.

**Evaluation strip below the four steps**

Speed depends on agreement and draft cost.

Local exact speculative-output matches: **194/240 (80.8%)**; a numerical tie change was reproduced in one case.

### Speaker notes

First, the drafter proposes a short continuation. Second, the target checks the proposed positions together. Third, for deterministic decoding, we accept the longest prefix agreeing with the target, correct the first disagreement, and discard later guesses. If all guesses agree, the target can add a bonus token. Fourth, we update the accepted context and reuse the cache for the next round.

We measured tokens per target call, which we call tau. It includes the correction or bonus token, so it is not an acceptance percentage. A simple approximation is speedup equals tau divided by one plus K times c, where c is the draft-to-target generation-cost ratio. Our calibration gave c about 0.67: drafting was relatively expensive. At K equals two, the approximation needs tau above 2.34 to win; English reached 2.16. The model is approximate and does not isolate Python overhead. We also found only 80.8 percent exact output matches. A targeted control reproduced one score tie caused by scoring multiple positions; other historical mismatches remain unexamined.

### Optional discussion formula - keep in notes if the slide is full

Approximate speedup = tau / (1 + K*c)

- Final calibration c = 0.668
- K=2 approximate break-even tau = 2.336
- English pooled tau = 2.157
- Predicted English speedup about 0.92x; observed about 0.92x
- Calibration includes prefill/bookkeeping; no causal overhead profiling was done

## Slide 6 - Wrap-up / Questions (15 seconds)

### Text for the slide

**One memorable sentence**

Fewer large-model calls improve speed only when drafting is cheap enough and the models agree.

**Questions?**

### Speaker notes

Fewer large-model calls improve speed only when drafting is cheap enough and the models agree. In this laptop benchmark, the saved target calls did not outweigh the added work. Thank you. Questions?

## Rubric coverage

| Evaluation criterion | Marks | Where this draft addresses it |
|---|---:|---|
| Part A: Concept understanding and comparative analysis | 2 | Slide 3 defines normal versus speculative decoding; slide 4 reports the measured comparison |
| Part A: Evaluation and presentation | 2 | Slides 4-5 distinguish speed, target-call efficiency, numerical correctness and study limitations |
| Part A: Similarity and AI | 1 | Cite sources and explain your own recorded methods/results; do not promise a detector score |
| Part B: Problem understanding and solution design | 3 | Slide 4 defines the translation latency problem; slides 3 and 5 describe source conditioning and solution design |
| Part B: Use of techniques and presentation | 3 | Slide 5 explains the four steps; slide 4 attributes the published case outcome |
| Part B: Similarity and AI | 1 | Identify the translation case as published research and cite it; do not claim it as an original experiment |

The rubric's Excellent band states similarity below 10% and "AI below 20% or *"; the asterisk is not explained in the supplied plan. No wording draft can certify those scores. Cite sources, retain your experimental evidence, understand the explanation, and follow faculty guidance on allowed assistance.

## Q&A preparation

**Did your experiment fail?**

The mechanism reduced target calls, but every tested K lost median throughput in this setup. The comparison answered the efficiency question and identified a language difference in tau.

**Is the Urdu effect universal?**

No. It is a result for this model pair and these ten questions. Drafter-target agreement, tokenization, task and target behavior can all matter.

**Why is Hindi slower if its tau matches English?**

Tau counts tokens per target call. It does not measure the cost of each call, drafting, cache work or verification. Similar counts need not mean equal time.

**Did you run machine translation yourself?**

No. The MT case is a cited research study. My original experiment measures multilingual text generation.

**Are outputs always identical?**

The algorithm's mathematical equivalence does not establish bit-exact behavior in this FP16 implementation. We measured 80.8% exact matches overall and directly investigated one numerical tie change.

**Did you prove Python overhead caused the slowdown?**

No. Drafting was expensive relative to the target in end-to-end calibration, but no profiler separated the contributing costs.

**Could another draft length win?**

Possibly. K=1 was not measured historically. State that no tested K won, not that every possible K must lose.

**How would you evaluate a new translation implementation?**

Use the same source sentences, target checkpoint and decoding settings for both modes. Compare latency/throughput and target calls, verify exact matches for deterministic decoding, and assess translation adequacy against references with bilingual review. This is a proposed extension, not an experiment or a quality score measured in this project.

## References and evidence

1. Leviathan, Y., Kalman, M. and Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML. https://proceedings.mlr.press/v202/leviathan23a.html . Translation setup/results: section 4.1 and Table 2.
2. Yi et al. (2024). Towards Fast Multilingual LLM Inference: Speculative Decoding and Specialized Drafters. EMNLP. https://aclanthology.org/2024.emnlp-main.602/ . Optional background, not an additional reproduced experiment.
3. Local final-run measurements: results_rp105/raw.csv and summary.json; original executed notebook: results_rp105/benchmark_executed.ipynb.
4. Data review and sensitivity checks: DATA_REVIEW.md and analysis/audit.json. Correctness diagnostic: analysis/correctness_probe.json and correctness_probe_logits.pt.
5. Supplied presentation template and NLP evaluation plan. CA3 is listed as 12 marks (CO4:5, CO5:7), with tentative date 22 October 2026. The plan's CA2 scaling figures disagree between pages; that does not change the stated CA3 total.

Use short citations in the existing slide footers or speaker notes. A seventh references slide would change the supplied six-page structure.
