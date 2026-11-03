"""Unzip Spider + BIRD Mini-Dev, build prompts, write data/*.jsonl.

    python prepare_data.py        (expects data/raw/spider.zip and data/raw/minidev.zip)

Train examples are dropped when the gold query fails, returns no rows (any WHERE 1=0 would match),
or the prompt is over MAX_PROMPT_TOKENS. Eval sets are kept whole so scores stay comparable.
"""
import json
import sqlite3
import zipfile
from pathlib import Path

from transformers import AutoTokenizer

from sql_rewards import execute

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
MAX_PROMPT_TOKENS = 2048
DATA = Path("data")

SYSTEM = (
    "You are a SQLite expert. Given a database schema and a question, reason step by step inside "
    "<think>...</think>, then give one SQLite query that answers the question inside <sql>...</sql>."
)


def get_schema(db_path, n_rows=3):
    """CREATE TABLE statements plus a few example rows per table, so the model sees real value formats."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")
    parts = []
    tables = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    for name, ddl in tables.fetchall():
        cur = conn.execute(f'SELECT * FROM "{name}" LIMIT {n_rows}')
        rows = [" | ".join(str(v)[:40] for v in row) for row in cur.fetchall()]
        header = " | ".join(d[0] for d in cur.description)
        parts.append(f"{ddl.strip()};\n/* {len(rows)} example rows:\n{header}\n" + "\n".join(rows) + "\n*/")
    conn.close()
    return "\n\n".join(parts)


def make_prompt(schema, question, evidence=None):
    user = f"Database schema:\n{schema}\n\nQuestion: {question}"
    if evidence:
        user += f"\nHint: {evidence}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def build(name, questions_json, db_root, sql_key, is_train, tok):
    examples = json.loads(Path(questions_json).read_text(encoding="utf-8"))
    schemas, out, dropped = {}, [], {"gold_error": 0, "gold_empty": 0, "too_long": 0}
    for ex in examples:
        db_path = (Path(db_root) / ex["db_id"] / f"{ex['db_id']}.sqlite").as_posix()
        if db_path not in schemas:
            schemas[db_path] = get_schema(db_path)
        prompt = make_prompt(schemas[db_path], ex["question"], ex.get("evidence"))
        n_tokens = len(tok(tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)).input_ids)
        if is_train:
            try:
                rows = execute(db_path, ex[sql_key], timeout=60)
            except Exception:
                dropped["gold_error"] += 1
                continue
            if not rows:
                dropped["gold_empty"] += 1
                continue
            if n_tokens > MAX_PROMPT_TOKENS:
                dropped["too_long"] += 1
                continue
        out.append({"prompt": prompt, "gold_sql": ex[sql_key], "db_path": db_path, "db_id": ex["db_id"],
                    "question": ex["question"], "difficulty": ex.get("difficulty"), "n_tokens": n_tokens})
    with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    print(f"{name}: kept {len(out)}/{len(examples)}, dropped {dropped}, "
          f"max prompt {max(r['n_tokens'] for r in out)} tokens")


if __name__ == "__main__":
    for z in ("spider.zip", "minidev.zip"):
        with zipfile.ZipFile(DATA / "raw" / z) as zf:  # extractall strips absolute and ../ paths
            zf.extractall(DATA, [n for n in zf.namelist() if not n.startswith("__MACOSX")])
    tok = AutoTokenizer.from_pretrained(MODEL)
    spider, bird = DATA / "spider_data", DATA / "minidev" / "MINIDEV"
    build("spider_train", spider / "train_spider.json", spider / "database", "query", True, tok)
    build("spider_dev", spider / "dev.json", spider / "database", "query", False, tok)
    build("bird_minidev", bird / "mini_dev_sqlite.json", bird / "dev_databases", "SQL", False, tok)
