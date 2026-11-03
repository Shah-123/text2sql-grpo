"""Gradio demo: ask a question about a SQLite database, get the model's reasoning, its SQL and the result.

    python app.py                    # base model + GRPO adapter (outputs/grpo_weak, run 2)
    python app.py --adapter ""       # base model only

Queries run on a read-only connection, SELECT/WITH only, with a 5 s limit (see sql_rewards.execute).
"""
import argparse
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_data import get_schema, make_prompt
from sql_rewards import execute, extract_sql

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
p.add_argument("--adapter", default="outputs/grpo_weak")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

DBS = {f"{src} / {f.stem}": f for src, root in (("Spider", "data/spider_data/database"),
                                                ("BIRD", "data/minidev/MINIDEV/dev_databases"))
       for f in sorted(Path(root).glob("*/*.sqlite"))}
MAX_ROWS = 100


def answer(db_name, upload, question, hint):
    if not question.strip():
        raise gr.Error("Type a question first.")  # with no question the model invents one
    db = upload or DBS[db_name]
    schema = get_schema(db)
    text = tok.apply_chat_template(make_prompt(schema, question, hint), tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=1024, do_sample=False)
    output = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    think = re.search(r"<think>(.*?)</think>", output, re.S)
    sql = extract_sql(output, lenient=True)
    if not sql:
        return output, "", None, "No SQL in the answer.", schema
    try:
        rows = execute(db, sql)
    except Exception as e:
        return (think.group(1).strip() if think else output), sql, None, f"Query failed: {e}", schema
    status = f"{len(rows)} rows" + (f" (first {MAX_ROWS} shown)" if len(rows) > MAX_ROWS else "")
    return (think.group(1).strip() if think else output), sql, [list(r) for r in rows[:MAX_ROWS]], status, schema


with gr.Blocks(title="Text-to-SQL GRPO") as demo:
    gr.Markdown("# Text-to-SQL with GRPO\nQwen2.5-Coder-3B-Instruct + LoRA, trained with execution-based rewards "
                f"on Spider. Adapter: `{args.adapter or 'none (base model)'}`.")
    with gr.Row():
        with gr.Column():
            db = gr.Dropdown(list(DBS), value="Spider / concert_singer", label="Database")
            upload = gr.File(label="...or upload your own SQLite file", file_types=[".sqlite", ".db"], type="filepath")
            question = gr.Textbox(label="Question", lines=2)
            hint = gr.Textbox(label="Hint (optional, BIRD-style evidence)")
            run = gr.Button("Generate SQL", variant="primary")
        with gr.Column():
            sql = gr.Code(label="SQL", language="sql")
            status = gr.Markdown()
            result = gr.Dataframe(label="Result")
            reasoning = gr.Textbox(label="Reasoning", lines=6)
    with gr.Accordion("Schema shown to the model", open=False):
        schema = gr.Code(language="sql")
    gr.Examples(  # dev-set questions the GRPO model gets right
        [["Spider / concert_singer", "How many singers do we have?", ""],
         ["Spider / wta_1", "Which year had the most matches?", ""],
         ["BIRD / toxicology", "What is the most common bond type?", "most common bond type refers MAX(COUNT(bond_type))"],
         ["BIRD / superhero", "Please list all the superpowers of 3-D Man.",
          "3-D Man refers to superhero_name = '3-D Man'; superpowers refers to power_name"]],
        [db, question, hint])
    run.click(answer, [db, upload, question, hint], [reasoning, sql, result, status, schema])

if __name__ == "__main__":
    demo.launch()
