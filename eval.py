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
p.add_argument("--batch-size", type=int, default=16)
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
for start in range(0, len(order), args.batch_size):
    batch = [rows[i] for i in order[start : start + args.batch_size]]
    texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in batch]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
    for r, ids in zip(batch, out[:, enc.input_ids.shape[1] :]):
        r["output"] = tok.decode(ids, skip_special_tokens=True)
        r["pred_sql"] = extract_sql(r["output"], lenient=not args.strict)
        r["label"] = classify(r["output"], r["gold_sql"], r["db_path"], lenient=not args.strict)
    print(f"{min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

out_dir = Path("outputs/eval")
out_dir.mkdir(parents=True, exist_ok=True)
name = args.name or Path(args.data).stem
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
print(json.dumps(summary, indent=2))
