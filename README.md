# Text-to-SQL with GRPO (Qwen2.5-Coder-3B)

Reinforcement learning with execution-based rewards (GRPO) teaches Qwen2.5-Coder-3B-Instruct to write better SQLite
queries. Trained on Spider, tested on Spider dev and BIRD Mini-Dev. Everything runs on one 16 GB consumer GPU on
native Windows.

## Results

Execution accuracy (%), greedy decoding:

| Model | Spider dev | BIRD Mini-Dev | Simple | Moderate | Challenging |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B-Instruct | 64.5 | 32.6 | 52.0 | 25.6 | 21.6 |
| + GRPO, 500 steps (this repo) | **67.2** | **36.0** | 52.7 | 31.6 | 22.5 |

- **Spider dev: +2.7 points.** GRPO fixed 81 questions and broke 53; exact McNemar test p = 0.019.
- **BIRD Mini-Dev: +3.4 points**, mostly on moderate questions (+6.0). GRPO fixed 66 and broke 49; p = 0.135, so
  this gain is **not statistically significant** on 500 questions. No BIRD database is seen during training.
- Answers got about 40% shorter (Spider 876 → 514 characters, BIRD 1,261 → 706).

**How answers are scored:** the rows returned by the predicted and the gold query are compared as sets (BIRD's
official EX rule), with a 30 s limit per query. Spider dev is scored the same way, which is *not* Spider's official
test-suite metric, so the Spider numbers are not directly comparable with leaderboard numbers. Base-model answers are
parsed leniently (code fences and bare `SELECT`s count); GRPO answers only count inside `<sql>` tags. If anything, the
comparison favours the base model.

## Method

**Prompt.** The system message asks for reasoning in `<think>…</think>` and one SQLite query in `<sql>…</sql>`. The user
message holds every table's `CREATE TABLE` statement with 3 example rows, the question, and for BIRD the "evidence"
hint.

**Data.** Spider train: 5,928 of 7,000 questions kept (815 prompts over 2,048 tokens, 254 gold queries that return no
rows and 3 gold queries that fail were dropped). The eval sets are kept whole.

**Reward** ([sql_rewards.py](sql_rewards.py), standard library only):

| Outcome | Reward |
|---|---|
| Output is exactly `<think>…</think><sql>…</sql>` | +1.0 × weight 0.25 |
| Query returns the same rows as the gold query | 1.0 |
| Query runs but returns different rows | 0.1 |
| Error, timeout, no SQL, not a SELECT, or a hard-coded answer (no `FROM`) | 0 |

Queries run on read-only SQLite connections, only `SELECT`/`WITH` is accepted, stacked statements are refused, and a
progress handler stops runaway queries (3 s during training).

**Training** ([train_grpo.py](train_grpo.py), TRL `GRPOTrainer`):
- LoRA r=16, α=32 on all linear layers of the bf16 model
- 6 completions per question, 2 questions per step, temperature 1.0, up to 1,024 new tokens
- learning rate 5e-6, no KL penalty (β=0), TRL's default DAPO loss, cut-off completions masked out
- 500 steps = 1,000 questions (about 17% of the training set)
- RTX 5060 Ti 16 GB, native Windows, generation through transformers (vLLM has no Windows build):
  37.6 s/step, 5.2 h in total, 13.9 GB peak memory

Training metrics, averaged over 50-step windows (training questions, 100 per window, so a few points is noise):

| Steps | Correct | SQL error | No SQL | Format reward | Answer length (tokens) |
|---|---|---|---|---|---|
| 1–50 | 57.8% | 18.2% | 5.8% | 0.42 | 214 |
| 251–300 | 66.5% | 13.7% | 0.0% | 0.97 | 134 |
| 451–500 | 74.2% | 12.8% | 0.0% | 0.97 | 128 |

## Error analysis

Full report: [outputs/eval/error_analysis.md](outputs/eval/error_analysis.md) (made by
[error_analysis.py](error_analysis.py)).

- **Wrong column names are still the biggest failure.** "No such column" errors fell from 110 to 95 on Spider and from
  157 to 138 on BIRD, but they remain the most common error.
- **GRPO's fixes are often simpler queries.** In 36 of the 81 Spider fixes and 30 of the 66 BIRD fixes, the GRPO query
  has fewer JOINs than the base model's. In the other direction, 21 of the 53 Spider questions it broke use more JOINs
  than the base model's answer.
- **Where it helped:** Spider questions whose gold query has a subquery or set operation (+5.0) or one JOIN (+4.1);
  BIRD questions with ORDER BY (+13.2), subqueries (+7.6) and GROUP BY (+7.3).
- **Where it did not:** questions needing two or more JOINs (Spider −1.1, BIRD −1.0).
- Fewer answers selected the wrong number of columns on Spider (36 → 26).

## Demo

```bash
python app.py
```

Opens at http://localhost:7860. Pick a Spider or BIRD database (or upload a `.sqlite` file), type a question and an
optional hint. The app shows the SQL, the result table, the model's reasoning and the schema it was given. Queries run
read-only with a 5 s limit.

## Reproduce

Commands are for Windows PowerShell, where the project was built.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wandb login
```

Data (Spider comes from a Hugging Face mirror because the official Google Drive link was over its download quota):

```powershell
curl.exe -L --create-dirs -o data/raw/spider.zip https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip
curl.exe -L --create-dirs -o data/raw/minidev.zip https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip
python prepare_data.py
```

Train, evaluate, analyse:

```powershell
python train_grpo.py --run-name grpo_zero
python eval.py --data data/spider_dev.jsonl --name base_spider_dev
python eval.py --data data/bird_minidev.jsonl --name base_bird_minidev
python eval.py --data data/spider_dev.jsonl --adapter outputs/grpo_zero --strict --name grpo_zero_spider_dev
python eval.py --data data/bird_minidev.jsonl --adapter outputs/grpo_zero --strict --name grpo_zero_bird_minidev
python error_analysis.py
```

`python sql_rewards.py` runs the reward function's self-check.

| File | What it does |
|---|---|
| [prepare_data.py](prepare_data.py) | Unzips the datasets, builds prompts, filters training questions, writes `data/*.jsonl` |
| [sql_rewards.py](sql_rewards.py) | Safe SQL execution, answer classification, format and execution rewards |
| [train_grpo.py](train_grpo.py) | GRPO training with LoRA |
| [eval.py](eval.py) | Execution accuracy on a split; saves every prediction |
| [error_analysis.py](error_analysis.py) | Compares two eval runs question by question |
| [app.py](app.py) | Gradio demo |

## Limitations and next steps

- One training run with one seed. 500 steps covered only about 17% of the training questions.
- The BIRD gain is not statistically significant. A larger BIRD set would settle it.
- By the end of training, 43% of question groups gave all 6 answers the same reward, so they taught nothing. Harder
  questions, or dropping the ones the model always solves, would help a longer run.
- Training prompts are capped at 2,048 tokens, while BIRD prompts average 2,431. Training on longer schemas may help
  BIRD.
- No SFT warm-up stage yet. GRPO alone taught the output format (97% by the end of training), so the main reason for
  one is gone; whether it still improves accuracy is untested.
- The BIRD eval took 4 h at `--batch-size 16` because long prompts overflowed GPU memory into system RAM. A smaller
  batch size should be faster (not measured).
