"""Training set for the second GRPO run: drop questions run 1 always solved, list weak-spot questions twice.

    python build_weak_set.py        # reads outputs/grpo_zero/completions/, writes data/spider_train_weak.jsonl

Weak spots, from outputs/eval/error_analysis.md: gold queries with 2+ JOINs, GROUP BY, or a subquery / set operation.
In run 1, 43% of the other questions got 6/6 correct answers (zero advantage, nothing learned) vs 13% of these.
"""
import glob
import json
import re
from collections import defaultdict

import pyarrow.parquet as pq

_n = lambda pat, s: len(re.findall(pat, s, re.I))
weak = lambda s: _n(r"\bJOIN\b", s) >= 2 or _n(r"\bGROUP\s+BY\b", s) > 0 or _n(r"\bSELECT\b", s) > 1

correct = defaultdict(list)  # user message -> was each logged sample correct
for f in glob.glob("outputs/grpo_zero/completions/*.parquet"):
    for r in pq.read_table(f, columns=["prompt", "execution_reward"]).to_pylist():
        correct[r["prompt"][-1]["content"]].append(r["execution_reward"] == 1.0)
easy = {q for q, oks in correct.items() if all(oks)}

train = [json.loads(line) for line in open("data/spider_train.jsonl", encoding="utf-8")]
kept = [r for r in train if r["prompt"][-1]["content"] not in easy]
out = kept + [r for r in kept if weak(r["gold_sql"])]  # the trainer shuffles
with open("data/spider_train_weak.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in out)
print(f"dropped {len(train) - len(kept)} always-solved, {len(out) - len(kept)} weak-spot questions listed twice, "
      f"{len(out)} rows ({(len(out) - len(kept)) * 2 / len(out):.0%} of draws are weak-spot)")
