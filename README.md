# Text-to-SQL with GRPO (Qwen2.5-Coder-3B)

Reinforcement learning with execution-based rewards (GRPO) teaches Qwen2.5-Coder-3B-Instruct to write better SQLite
queries. Trained on Spider, tested on Spider dev and BIRD Mini-Dev. Everything runs on one 16 GB consumer GPU on
native Windows.

## Results

Execution accuracy (%), greedy decoding:

| Model | Spider dev | BIRD Mini-Dev | Simple | Moderate | Challenging |
|---|---|---|---|---|---|
| Qwen2.5-Coder-3B-Instruct | 64.5 | 32.6 | 52.0 | 25.6 | 21.6 |
| + GRPO run 1, 500 steps | 67.2 | 36.0 | 52.7 | 31.6 | 22.5 |
| + GRPO run 2, 500 more steps weighted toward run 1's weak spots | **67.6** | **38.2** | **58.1** | **32.8** | 22.5 |

Question by question, with an exact McNemar test (fixed / broke = questions that went from wrong to right / right to
wrong):

| Comparison | Spider dev: fixed / broke, p | BIRD Mini-Dev: fixed / broke, p |
|---|---|---|
| base → run 1 | 81 / 53, p = 0.019 | 66 / 49, p = 0.135 |
| base → run 2 | 95 / 63, p = 0.013 | 69 / 41, p = 0.010 |
| run 1 → run 2 | 47 / 43, p = 0.752 | 31 / 20, p = 0.161 |

- **Run 2 beats the base model on both sets: +3.1 points on Spider, +5.6 on BIRD.** Both gains are significant, also
  after a Bonferroni correction for the three comparisons per set (p × 3 = 0.039 and 0.030). No BIRD database is seen
  during training.
- **Run 2 vs. run 1 is not significant** (+0.4 and +2.2 points). Run 1's BIRD gain alone was not significant either.
- Answers got shorter with each run (Spider 877 → 514 → 441 characters, BIRD 1,264 → 710 → 546).

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

**Training, run 1** ([train_grpo.py](train_grpo.py), TRL `GRPOTrainer`):
- LoRA r=16, α=32 on all linear layers of the bf16 model
- 6 completions per question, 2 questions per step, temperature 1.0, up to 1,024 new tokens
- learning rate 5e-6, no KL penalty (β=0), TRL's default DAPO loss, cut-off completions masked out
- 500 steps = 1,000 questions (about 17% of the training set)
- RTX 5060 Ti 16 GB, native Windows, generation through transformers (vLLM has no Windows build):
  37.6 s/step, 5.2 h in total, 13.9 GB peak memory

**Run 2** keeps training run 1's adapter (`--init-adapter`, fresh optimizer and learning-rate schedule, same settings,
500 steps, about 36 s/step, 13.8 GB peak) on a set built by [build_weak_set.py](build_weak_set.py) from run 1's logged
completions. It drops the 299 questions where every one of run 1's samples was correct, then lists the 2,609
questions whose gold query has 2+ JOINs, GROUP BY, or a subquery or set operation twice. That gives 8,238 rows, with
weak-spot questions making up 63% of draws. In run 1, only 13% of these questions' groups had all 6 answers correct,
vs. 43% for the other questions.

Training metrics, averaged over 50-step windows (training questions, 100 per window, so a few points is noise; run 2
draws harder questions, so its "Correct" is not comparable with run 1's):

| Run | Steps | Correct | SQL error | No SQL | Format reward | Answer length (tokens) | All-same-reward groups |
|---|---|---|---|---|---|---|---|
| 1 | 1–50 | 57.8% | 18.2% | 5.8% | 0.42 | 214 | 2% |
| 1 | 251–300 | 66.5% | 13.7% | 0.0% | 0.97 | 134 | 34% |
| 1 | 451–500 | 74.2% | 12.8% | 0.0% | 0.97 | 128 | 43% |
| 2 | 1–50 | 67.3% | 13.3% | 0.0% | 0.97 | 132 | 31% |
| 2 | 251–300 | 67.8% | 13.0% | 0.2% | 0.99 | 123 | 36% |
| 2 | 451–500 | 61.3% | 12.2% | 0.2% | 0.98 | 119 | 38% |

"All-same-reward groups" are questions where all 6 answers got the same reward, so GRPO learns nothing from them.
The weighted set kept this below run 1's 43%, but not by much.

## Error analysis

Full report: [outputs/eval/error_analysis.md](outputs/eval/error_analysis.md) (made by
[error_analysis.py](error_analysis.py)).

- **Wrong column names are still the biggest failure.** "No such column" errors went 110 → 95 → 100 on Spider and
  157 → 138 → 138 on BIRD (base → run 1 → run 2), and remain the most common error.
- **Run 1's fixes are often simpler queries.** In 36 of the 81 Spider fixes and 30 of the 66 BIRD fixes, the run 1
  query has fewer JOINs than the base model's. In the other direction, 21 of the 53 Spider questions it broke use more
  JOINs than the base model's answer.
- **Where run 1 helped:** Spider questions whose gold query has a subquery or set operation (+5.0) or one JOIN (+4.1);
  BIRD questions with ORDER BY (+13.2), subqueries (+7.6) and GROUP BY (+7.3).
- **Run 2 did not fix the weak spots it was aimed at.** Questions needing two or more JOINs did not move at all
  (Spider 44.3 → 43.2 → 43.2%, BIRD 17.5 → 16.5 → 16.5%). Against run 1, GROUP BY went −1.4 on Spider and +9.1 on BIRD
  (5 of 55 questions), and subqueries +3.8 on Spider and 0 on BIRD. Its BIRD gain over run 1 came mostly from simple
  questions (+5.4).
- Fewer answers selected the wrong number of columns on Spider (36 → 26 → 22).

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
python build_weak_set.py
python train_grpo.py --run-name grpo_weak --init-adapter outputs/grpo_zero --data data/spider_train_weak.jsonl
python eval.py --data data/spider_dev.jsonl --name base_spider_dev
python eval.py --data data/bird_minidev.jsonl --name base_bird_minidev
python eval.py --data data/spider_dev.jsonl --adapter outputs/grpo_zero --strict --name grpo_zero_spider_dev
python eval.py --data data/bird_minidev.jsonl --adapter outputs/grpo_zero --strict --name grpo_zero_bird_minidev
python eval.py --data data/spider_dev.jsonl --adapter outputs/grpo_weak --strict --name grpo_weak_spider_dev
python eval.py --data data/bird_minidev.jsonl --adapter outputs/grpo_weak --strict --name grpo_weak_bird_minidev
python error_analysis.py
```

If training is interrupted, run the same command with `--resume`. It continues from the last checkpoint (every 50
steps). If an eval is interrupted, run the same command again: it picks up from `outputs/eval/<name>.partial.jsonl`.

`python sql_rewards.py` runs the reward function's self-check.

| File | What it does |
|---|---|
| [prepare_data.py](prepare_data.py) | Unzips the datasets, builds prompts, filters training questions, writes `data/*.jsonl` |
| [sql_rewards.py](sql_rewards.py) | Safe SQL execution, answer classification, format and execution rewards |
| [train_grpo.py](train_grpo.py) | GRPO training with LoRA |
| [build_weak_set.py](build_weak_set.py) | Run 2's training set, weighted toward run 1's weak spots |
| [eval.py](eval.py) | Execution accuracy on a split; saves every prediction |
| [error_analysis.py](error_analysis.py) | Compares the eval runs question by question |
| [app.py](app.py) | Gradio demo |

## Limitations and next steps

- One seed per run. Run 1's 500 steps covered only about 17% of the training questions.
- There is no control run with 500 more steps on the unweighted set. So even if run 2's gain over run 1 were
  significant, it could come from the extra training and not from the weighting.
- Weighting toward hard questions did not lift 2+ JOIN accuracy. A different signal (for example, partial credit for
  the right tables) may be needed there.
- **Eval noise:** in bf16, greedy outputs depend on which questions share a batch. Between batch size 8 and 16, 8 of
  16 BIRD answers changed text and 4 changed label. Every model here was evaluated at `--batch-size 16`, but
  differences of a few questions are within this noise.
- Training prompts are capped at 2,048 tokens, while BIRD prompts average 2,431. Training on longer schemas may help
  BIRD.
- No SFT warm-up stage yet. GRPO alone taught the output format (97% by the end of training), so the main reason for
  one is gone; whether it still improves accuracy is untested.
- The BIRD eval takes 3–4 h at `--batch-size 16` because long prompts overflow GPU memory into system RAM. A smaller
  batch would be faster, but it changes the outputs (see eval noise above), so every model would need to be
  re-evaluated with it.
