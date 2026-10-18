"""Targeted, untimed output-equivalence probe using cached local models.

Records actual baseline/assisted IDs and compares target scores at their first
divergence. This is not a replacement benchmark or a universal correctness proof.
"""
import json
import copy
import platform
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from benchmark import GREEDY, set_draft_strategy

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis"
TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    target_path = snapshot_download(TARGET, local_files_only=True)
    draft_path = snapshot_download(DRAFT, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    print("Loading cached target and draft for an untimed probe", flush=True)
    target = AutoModelForCausalLM.from_pretrained(target_path, local_files_only=True,
                                                dtype=torch.float16, device_map="cuda").eval()
    draft = AutoModelForCausalLM.from_pretrained(draft_path, local_files_only=True,
                                               dtype=torch.float16, device_map="cuda").eval()
    prompt = json.loads((HERE / "prompts.json").read_text(encoding="utf-8"))["en"][5]
    chat = tok.apply_chat_template([dict(role="system", content="Answer in the same language as the question."),
                                   dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to("cuda")
    prompt_ids = inputs["input_ids"][0].tolist()
    records = []

    def capture(model, args, kwargs, output):
        logits = output.logits.detach().cpu()
        positions = kwargs.get("cache_position")
        if positions is None:
            raise RuntimeError("No cache_position: cannot map scores to generated token positions")
        positions = positions.detach().cpu().tolist()[-logits.shape[1]:]
        ids = kwargs["input_ids"].detach().cpu()[0].tolist()
        records.append((positions, ids, logits[0]))

    kwargs = dict(max_new_tokens=128, pad_token_id=tok.eos_token_id, repetition_penalty=1.05, **GREEDY)

    def run(k, collect):
        records.clear()
        handle = target.register_forward_hook(capture, with_kwargs=True) if collect else None
        try:
            if k:
                set_draft_strategy(draft, k)
            ids = target.generate(**inputs, **kwargs, **({"assistant_model": draft} if k else {}))
            result = ids[0, len(prompt_ids):].tolist()
            snapshot = list(records)
            return result, snapshot
        finally:
            if handle:
                handle.remove()

    base, base_records = run(0, True)
    base_repeat, _ = run(0, False)
    spec, spec_records = run(4, True)
    spec_repeat, _ = run(4, False)
    first = next((i for i, (a, b) in enumerate(zip(base, spec)) if a != b), None)
    if first is None and len(base) != len(spec):
        first = min(len(base), len(spec))
    report = dict(prompt=prompt, lang="en", prompt_id=5, k=4, rep_penalty=1.05,
                  max_new=128, target=TARGET, draft=DRAFT, dtype="float16",
                  gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                  torch=torch.__version__, transformers=transformers.__version__,
                  target_revision=getattr(target.config, "_commit_hash", None),
                  draft_revision=getattr(draft.config, "_commit_hash", None),
                  baseline_repeat_matches=base == base_repeat,
                  speculative_repeat_matches=spec == spec_repeat,
                  matches_baseline=base == spec, first_divergence_token_0based=first,
                  baseline_ids=base, speculative_ids=spec,
                  baseline_repeat_ids=base_repeat, speculative_repeat_ids=spec_repeat,
                  baseline_text=tok.decode(base, skip_special_tokens=True),
                  speculative_text=tok.decode(spec, skip_special_tokens=True))
    if first is not None and first < min(len(base), len(spec)):
        position = len(prompt_ids) + first - 1
        def at_position(calls):
            for positions, ids, logits in reversed(calls):
                if position in positions:
                    return logits[positions.index(position)]
            raise RuntimeError("Divergence scoring position missing from trace")
        raw_a, raw_b = at_position(base_records), at_position(spec_records)
        prefix = torch.tensor([prompt_ids + base[:first]], dtype=torch.long)
        processor = RepetitionPenaltyLogitsProcessor(1.05)
        a = processor(prefix, raw_a.float().unsqueeze(0).clone())[0]
        b = processor(prefix, raw_b.float().unsqueeze(0).clone())[0]
        aid, bid = base[first], spec[first]
        def top(scores):
            values, ids = torch.topk(scores, 5)
            return [dict(id=i, token=tok.decode([i]), score=v) for i, v in zip(ids.tolist(), values.tolist())]
        report["scores"] = dict(
            baseline_top5=top(a), speculative_top5=top(b),
            baseline_selected_id=aid, speculative_selected_id=bid,
            reconstructed_baseline_argmax=int(a.argmax()), reconstructed_speculative_argmax=int(b.argmax()),
            argmax_reconstruction_matches=(int(a.argmax()) == aid and int(b.argmax()) == bid),
            raw_max_abs_difference=float((raw_a.float() - raw_b.float()).abs().max()),
            baseline_margin_a_minus_b=float(a[aid] - a[bid]),
            speculative_margin_a_minus_b=float(b[aid] - b[bid]))
        # Control: identical prefix, weights and freshly built KV-cache contents.
        # Only the number of scored new positions differs; no draft model or
        # assisted-generation cache rollback is involved in these two forwards.
        device_prefix = prefix.to(target.device)
        prefill = target(input_ids=device_prefix[:, :-1], use_cache=True)
        cache = prefill.past_key_values
        single_ids = device_prefix[:, -1:]
        block_ids = torch.tensor([[int(single_ids[0, 0])] + spec[first:first + 4]], device=target.device)
        single = target(input_ids=single_ids, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, 0].detach().cpu()
        block = target(input_ids=block_ids, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0, 0].detach().cpu()
        one_scores = processor(prefix, single.float().unsqueeze(0).clone())[0]
        block_scores = processor(prefix, block.float().unsqueeze(0).clone())[0]
        report["same_cache_shape_control"] = dict(
            single_top5=top(one_scores), block_top5=top(block_scores),
            single_argmax=int(one_scores.argmax()), block_argmax=int(block_scores.argmax()),
            rank_changed=int(one_scores.argmax()) != int(block_scores.argmax()),
            single_margin_a_minus_b=float(one_scores[aid] - one_scores[bid]),
            block_margin_a_minus_b=float(block_scores[aid] - block_scores[bid]),
            max_abs_raw_difference=float((single.float() - block.float()).abs().max()))
        torch.save(dict(baseline_raw=raw_a, speculative_raw=raw_b, shared_prefix=prefix,
                        baseline_processed=a, speculative_processed=b,
                        control_single_raw=single, control_block_raw=block), OUT / "correctness_probe_logits.pt")
    OUT.mkdir(exist_ok=True)
    (OUT / "correctness_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_ids") and not k.endswith("_text")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
