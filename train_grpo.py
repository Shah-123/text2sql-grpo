"""GRPO on Spider train with execution rewards. LoRA on a bf16 model; generation runs through transformers
(vLLM has no native Windows build).

    python train_grpo.py --run-name grpo_zero                          # GRPO only ("R1-Zero" ablation)
    python train_grpo.py --run-name grpo_sft --model outputs/sft_merged  # after the SFT warm-up
    python train_grpo.py --run-name timing --max-steps 10              # speed / memory check
    python train_grpo.py --run-name grpo_zero --resume                 # continue from the last checkpoint
    python train_grpo.py --run-name grpo_weak --init-adapter outputs/grpo_zero --data data/spider_train_weak.jsonl
"""
import argparse
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import sql_rewards
from sql_rewards import execution_reward, format_reward

p = argparse.ArgumentParser()
p.add_argument("--run-name", required=True)
p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
p.add_argument("--data", default="data/spider_train.jsonl")
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--lr", type=float, default=5e-6)
p.add_argument("--num-generations", type=int, default=6, help="group size; 4 if out of memory")
p.add_argument("--prompts-per-step", type=int, default=2)
p.add_argument("--max-completion-length", type=int, default=1024)
p.add_argument("--continuous-batching", action=argparse.BooleanOptionalAction, default=True,
               help="transformers' paged generation engine: 49 vs 63 s/step and 13.2 vs 15.8 GB peak on a 5060 Ti")
p.add_argument("--resume", action="store_true")
p.add_argument("--init-adapter", help="keep training this LoRA adapter (fresh optimizer) instead of a new one")
args = p.parse_args()

os.environ.setdefault("WANDB_PROJECT", "text2sql-grpo")
for stream in (sys.stdout, sys.stderr):  # TRL's sample table crashed on cp1252 (Windows default) at step 22
    stream.reconfigure(encoding="utf-8", errors="replace")
sql_rewards.TIMEOUT = 3.0  # Spider gold queries run in milliseconds; this only stops runaway attempts

class Bf16Up(TrainerCallback):
    """TRL's continuous-batching generation casts the whole model, LoRA included, to bf16 in place, so the optimizer
    state ends up bf16. On --resume the Trainer reloads the adapter (and so the optimizer state) as fp32 first, and the
    first optimizer step then fails on mixed dtypes. Do TRL's cast up front, optimizer state included."""

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        model.to(torch.bfloat16)
        for s in getattr(optimizer, "optimizer", optimizer).state.values():
            for k, v in s.items():
                if k != "step" and torch.is_tensor(v) and v.is_floating_point():
                    s[k] = v.to(torch.bfloat16)


tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
if args.init_adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)

config = GRPOConfig(
    output_dir=f"outputs/{args.run_name}",
    run_name=args.run_name,
    report_to="wandb",
    max_steps=args.max_steps,
    learning_rate=args.lr,
    num_generations=args.num_generations,
    per_device_train_batch_size=args.num_generations,  # one question's group per micro-batch
    gradient_accumulation_steps=args.prompts_per_step,  # questions per optimizer step
    temperature=1.0,  # variety inside a group is what GRPO learns from
    max_completion_length=args.max_completion_length,
    beta=0.0,  # raise to ~0.02 if outputs drift into nonsense
    mask_truncated_completions=True,  # don't learn from cut-off rambling
    reward_weights=[0.25, 1.0],  # format_reward, execution_reward
    use_transformers_continuous_batching=args.continuous_batching,
    gradient_checkpointing=True,
    bf16=True,
    logging_steps=1,
    save_steps=50,
    log_completions=True,  # read some every ~50 steps: metrics can look fine while the model does something silly
    num_completions_to_print=1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=[format_reward, execution_reward],
    args=config,
    callbacks=[Bf16Up()],
    train_dataset=load_dataset("json", data_files=args.data, split="train"),
    peft_config=None if args.init_adapter else LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"),
)
result = trainer.train(resume_from_checkpoint=args.resume or None)
trainer.save_model()
print(f"steps={result.global_step} runtime={result.metrics['train_runtime']:.0f}s "
      f"s/step={result.metrics['train_runtime'] / result.global_step:.1f} "
      f"peak_gpu_mem={torch.cuda.max_memory_reserved() / 2**30:.1f}GB")
