"""Execution accuracy of a model on a prepared split. Writes every prediction so errors can be compared later.

    python eval.py --data data/spider_dev.jsonl --name base_spider_dev            # base model, lenient parsing
    python eval.py --data data/spider_dev.jsonl --adapter outputs/grpo --strict   # trained model, tags required

Lenient parsing also accepts ```sql fences / bare SELECTs: the base model was never taught the tags,
so strict scoring would put it near 0% and make the improvement look fake.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sql_rewards
from sql_rewards import classify, extract_sql

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--name", help="output file name (default: data file stem)")
p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
p.add_argument("--adapter", help="LoRA adapter dir")
p.add_argument("--strict", action="store_true", help="require <sql> tags")
p.add_argument("--limit", type=int)
p.add_argument("--batch-size", type=int, default=16,
               help="keep it equal across models you compare: in bf16, batch shape changes greedy outputs (8 of 16 "
                    "BIRD answers differed between batch 8 and 16). 16 x 5k-token BIRD prompts spill out of 16 GB VRAM.")
p.add_argument("--max-new-tokens", type=int, default=1024)
p.add_argument("--timeout", type=float, default=30.0, help="seconds per query; 30 = BIRD's official value")
args = p.parse_args()
sql_rewards.TIMEOUT = args.timeout

tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

rows = [json.loads(line) for line in open(args.data, encoding="utf-8")][: args.limit]
order = sorted(range(len(rows)), key=lambda i: rows[i]["n_tokens"])  # similar lengths per batch = less padding
batches = [order[s : s + args.batch_size] for s in range(0, len(order), args.batch_size)]

out_dir = Path("outputs/eval")
out_dir.mkdir(parents=True, exist_ok=True)
name = args.name or Path(args.data).stem
partial = out_dir / f"{name}.partial.jsonl"  # one line per finished question; rerun the same command to resume
done = {d.pop("i"): d for d in map(json.loads, open(partial, encoding="utf-8"))} if partial.exists() else {}
if done:
    print(f"resuming: {len(done)}/{len(rows)} already done", flush=True)
with open(partial, "a", encoding="utf-8") as log:
    for batch in batches:
        if all(i in done for i in batch):
            continue
        texts = [tok.apply_chat_template(rows[i]["prompt"], tokenize=False, add_generation_prompt=True) for i in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        for i, ids in zip(batch, out[:, enc.input_ids.shape[1] :]):
            text = tok.decode(ids, skip_special_tokens=True)
            r = rows[i]
            done[i] = {"output": text, "pred_sql": extract_sql(text, lenient=not args.strict),
                       "label": classify(text, r["gold_sql"], r["db_path"], lenient=not args.strict)}
            log.write(json.dumps({"i": i, **done[i]}) + "\n")
        log.flush()
        print(f"{len(done)}/{len(rows)}", flush=True)
for i, r in enumerate(rows):
    r.update(done[i])

with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps({k: v for k, v in r.items() if k != "prompt"}) + "\n" for r in rows)

by_diff = defaultdict(list)
for r in rows:
    by_diff["all"].append(r["label"] == "correct")
    if r["difficulty"]:
        by_diff[r["difficulty"]].append(r["label"] == "correct")
summary = {
    "data": args.data, "model": args.model, "adapter": args.adapter, "strict": args.strict, "n": len(rows),
    "accuracy": {k: round(100 * sum(v) / len(v), 1) for k, v in by_diff.items()},
    "labels": dict(Counter(r["label"] for r in rows)),
}
(out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
partial.unlink()
print(json.dumps(summary, indent=2))
