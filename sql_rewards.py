"""Execution-based rewards for text-to-SQL GRPO. Standard library only.

Run `python sql_rewards.py` for the self-check on a toy database.
"""
import functools
import re
import sqlite3
import time
from pathlib import Path

TIMEOUT = 5.0  # seconds per query; set sql_rewards.TIMEOUT = 2-3 during training to keep steps fast

_NO_TAG = r"(?:(?!</?(?:think|sql)>).)+"  # text that does not open/close another tag
_FORMAT = re.compile(rf"\s*<think>{_NO_TAG}</think>\s*<sql>{_NO_TAG}</sql>\s*", re.S | re.I)
_SQL_TAG = re.compile(r"<sql>(.*?)</sql>", re.S | re.I)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)
_BARE = re.compile(r"(\b(?:SELECT|WITH)\b[^;]*)", re.S | re.I)
_FROM = re.compile(r"\bFROM\b", re.I)

EXEC_SCORE = {"correct": 1.0, "wrong_result": 0.1}  # everything else scores 0.0


def extract_sql(text, lenient=False):
    """SQL inside <sql> tags. lenient=True also takes a ```sql fence or a bare SELECT (for the untrained base model)."""
    for pat in (_SQL_TAG, _FENCE, _BARE) if lenient else (_SQL_TAG,):
        m = pat.search(text)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return None


def execute(db_path, sql, timeout=None):
    """Run one SELECT/WITH query on a read-only connection. Raises on error, timeout or a non-SELECT statement."""
    if not re.match(r"\s*(?:SELECT|WITH)\b", sql, re.I):
        raise ValueError("only SELECT/WITH queries are allowed")
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode(errors="replace")  # some BIRD text is not valid UTF-8
    deadline = time.monotonic() + (timeout or TIMEOUT)
    conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)  # non-zero return aborts the query
    try:
        return conn.execute(sql).fetchall()  # sqlite3 refuses stacked statements ("SELECT 1; DROP ...")
    finally:
        conn.close()


@functools.lru_cache(maxsize=4096)
def _gold_rows(db_path, gold_sql):
    return frozenset(execute(db_path, gold_sql))


def classify(text, gold_sql, db_path, lenient=False):
    """'correct' | 'wrong_result' | 'exec_error' | 'hardcoded' | 'no_sql' | 'gold_error'.

    Results are compared as sets of rows (row order and duplicates ignored), like BIRD's official EX.
    A gold query that fails or times out scores 0, also like BIRD. prepare_data.py keeps such examples
    out of the training set.
    """
    sql = extract_sql(text, lenient)
    if not sql:
        return "no_sql"
    if not _FROM.search(sql):  # answer worked out in <think> and typed in, e.g. SELECT 'Ayesha'
        return "hardcoded"
    try:
        gold = _gold_rows(db_path, gold_sql)
    except Exception:
        return "gold_error"
    try:
        pred = frozenset(execute(db_path, sql))
    except Exception:
        return "exec_error"
    return "correct" if pred == gold else "wrong_result"


def _text(completion):  # TRL passes a str, or [{"role": "assistant", "content": ...}] for chat data
    return completion if isinstance(completion, str) else completion[-1]["content"]


def format_reward(completions, **kwargs):
    """1.0 if the output is exactly <think>...</think><sql>...</sql>."""
    return [float(bool(_FORMAT.fullmatch(_text(c)))) for c in completions]


def execution_reward(completions, gold_sql, db_path, log_metric=None, **kwargs):
    """1.0 correct rows, 0.1 runs but wrong rows, 0.0 otherwise. gold_sql/db_path are dataset columns."""
    labels = [classify(_text(c), g, d) for c, g, d in zip(completions, gold_sql, db_path)]
    if log_metric:  # passed by TRL's GRPOTrainer; shows up in W&B next to the built-in metrics
        for name in ("correct", "wrong_result", "exec_error", "hardcoded", "no_sql"):
            log_metric(f"sql/{name}_frac", labels.count(name) / len(labels))
    return [EXEC_SCORE.get(label, 0.0) for label in labels]


if __name__ == "__main__":
    import tempfile

    folder = Path(tempfile.mkdtemp()) / "path with spaces"
    folder.mkdir()
    db = str(folder / "toy.sqlite")
    with sqlite3.connect(db) as c:
        c.executescript("""
            CREATE TABLE singer(id INTEGER PRIMARY KEY, name TEXT, age INT, country TEXT);
            INSERT INTO singer VALUES (1,'Ayesha',30,'PK'), (2,'Bilal',25,'PK'), (3,'Chen',40,'CN');
        """)
    gold = "SELECT name FROM singer WHERE country = 'PK'"
    TIMEOUT = 1.0

    cases = [  # (sql, expected reward, why)
        ("SELECT s.name FROM singer AS s WHERE s.age < 35 ORDER BY s.name DESC", 1.0, "different SQL, same rows"),
        ("SELECT DISTINCT name FROM singer WHERE country IN ('PK')", 1.0, "another rewrite"),
        ("SELECT name FROM singer", 0.1, "runs, wrong rows"),
        ("SELECT nme FROM singer", 0.0, "error"),
        ("SELECT 'Ayesha' UNION SELECT 'Bilal'", 0.0, "hard-coded answer"),
        ("DROP TABLE singer", 0.0, "not a SELECT"),
        (f"{gold}; DROP TABLE singer", 0.0, "stacked statements"),
        ("WITH x AS (SELECT 1) DELETE FROM singer", 0.0, "write hidden behind WITH"),
        ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c", 0.0, "runaway"),
    ]
    for sql, want, why in cases:
        start = time.monotonic()
        got = execution_reward([f"<think>plan</think><sql>{sql}</sql>"], [gold], [db])[0]
        assert got == want, f"{why}: got {got}, want {want}"
        assert time.monotonic() - start < 3, f"{why}: timeout not enforced"
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM singer").fetchone()[0] == 3, "database was modified"

    chat = [{"role": "assistant", "content": f"<think>x</think>\n<sql>{gold}</sql>"}]
    assert execution_reward([chat], [gold], [db]) == [1.0], "chat-format completion"
    assert format_reward(["<think>a</think>\n<sql>SELECT 1</sql>\n", chat]) == [1.0, 1.0]
    assert format_reward(["SELECT 1", "<sql>SELECT 1</sql>", "<think>a</think><sql>x</sql> extra",
                          "<think>a</think><sql>x</sql><sql>y</sql>"]) == [0.0] * 4

    fenced = f"Here you go:\n```sql\n{gold}\n```"
    assert classify(fenced, gold, db) == "no_sql"
    assert classify(fenced, gold, db, lenient=True) == "correct"
    assert classify(f"The answer is {gold};", gold, db, lenient=True) == "correct"
    assert classify(f"<sql>{gold}</sql>", "SELECT nme FROM singer", db) == "gold_error"
    print("sql_rewards: all checks passed")
