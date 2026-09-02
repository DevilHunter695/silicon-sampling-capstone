"""Track B: QLoRA fine-tune on real WVS-7 India respondents, out-of-fold.

Designed to run standalone on the university AI Lab's H100 machines --
not a notebook, so it survives disconnects (run under `nohup` or `tmux`).

Trains on ONE fold's training respondents (default: fold 0), then predicts
on that fold's held-out test respondents, using the EXACT SAME prompt
construction (verbalize_demographics / verbalize_item / build_prompt) and
the EXACT SAME item set as Track A's zero-shot runs, so this is a genuine
apples-to-apples "does fine-tuning help" comparison, not a different
experiment wearing the same name. Output parquet matches Track A's schema
so it drops straight into `src.report.evaluate_run` / `src.eval.subgroups`.

Usage (from repo root, after `pip install -r scripts/trackB_requirements.txt`):
    python -m scripts.trackB_finetune --smoke-test          # ~5 min correctness check
    python -m scripts.trackB_finetune --fold 0               # real run, fold 0, 15 items
    python -m scripts.trackB_finetune --fold 0 --n-items 144  # full item battery (slow)

Prints a final summary block (accuracy, n predictions) at the end -- paste
that back for evaluation once the run finishes.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PROCESSED, RESULTS_DIR
from src.prompts.templates import build_prompt
from src.prompts.verbalize import load_codebook, verbalize_demographics, verbalize_item
from src.inference.prompting import build_answer_instruction, parse_answer_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trackB_run.log")],
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"  # same base model Track A tested zero-shot via Groq


def build_examples(df: pd.DataFrame, respondent_ids: list, selected_items: list, codebook: dict) -> list:
    """One training example per (respondent, item): prompt text + the correct answer digit,
    built from the identical pipeline Track A used -- codebook-grounded question wording,
    codebook-grounded demographic verbalization, same P2 condition, same closing instruction."""
    examples = []
    sub = df[df["respondent_id"].isin(respondent_ids)]
    for _, row in sub.iterrows():
        demo = verbalize_demographics(row, codebook)
        for question_id in selected_items:
            item = verbalize_item(question_id, codebook)
            true_code = row.get(question_id)
            if pd.isna(true_code):
                continue
            true_code = int(true_code)
            if true_code not in item["code_to_index"]:
                continue
            option_labels = [str(c) for c in item["ordinal_values"]]
            prompt = build_prompt(
                "P2", item["question_text"], item["options_text"], **demo
            ) + build_answer_instruction(option_labels)
            answer = str(true_code)
            examples.append({"text": prompt + "\n\n" + answer, "prompt": prompt, "answer": answer})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=15, help="First N of the 144 selected items (15 matches Track A's widened run)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--smoke-test", action="store_true", help="Tiny subset (5 train, 5 test respondents, 3 items) -- run this FIRST to catch errors before spending real GPU time")
    ap.add_argument("--output-dir", default="/tmp/trackB_output")
    args = ap.parse_args()

    logger.info(f"CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        logger.error("No CUDA device found -- this must run on the lab's GPU machine, not a laptop.")
        sys.exit(1)

    df = pd.read_parquet(DATA_PROCESSED / "ind_wvs7.parquet")
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))
    with open(DATA_PROCESSED / "selected_items.json") as f:
        selected_items = json.load(f)["selected_items"][: args.n_items]
    with open(DATA_PROCESSED / "folds.json") as f:
        folds = json.load(f)["folds"]
    codebook = load_codebook()

    fold = folds[args.fold]
    train_ids, test_ids = fold["train"], fold["test"]
    if args.smoke_test:
        train_ids, test_ids = train_ids[:5], test_ids[:5]
        selected_items = selected_items[:3]
        logger.info("SMOKE TEST: 5 train respondents, 5 test respondents, 3 items")

    logger.info(f"Fold {args.fold}: {len(train_ids)} train respondents, {len(test_ids)} test respondents, {len(selected_items)} items")

    logger.info("Building training examples (same prompt pipeline as Track A)...")
    train_examples = build_examples(df, train_ids, selected_items, codebook)
    logger.info(f"Built {len(train_examples)} training examples")
    if not train_examples:
        logger.error("No training examples built -- check data/config paths.")
        sys.exit(1)

    # ---- Model + QLoRA setup ----
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    logger.info(f"Loading {args.model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=20,
        learning_rate=args.lr,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        seed=42,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_examples,
        args=training_args,
        packing=False,
        max_seq_length=768,
        tokenizer=tokenizer,
        formatting_func=lambda x: x["text"],
    )

    logger.info("Starting training...")
    t0 = time.time()
    trainer.train()
    logger.info(f"Training done in {(time.time()-t0)/60:.1f} min")

    model_path = f"{args.output_dir}/model_fold_{args.fold}"
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Saved fine-tuned model to {model_path}")

    # ---- Out-of-fold prediction on test respondents ----
    logger.info("Running out-of-fold inference on test respondents...")
    model.eval()
    rows = []
    test_sub = df[df["respondent_id"].isin(test_ids)]
    for _, row in test_sub.iterrows():
        demo = verbalize_demographics(row, codebook)
        for question_id in selected_items:
            item = verbalize_item(question_id, codebook)
            true_code = row.get(question_id)
            if pd.isna(true_code):
                continue
            true_code = int(true_code)
            if true_code not in item["code_to_index"]:
                continue
            option_labels = [str(c) for c in item["ordinal_values"]]
            prompt = build_prompt("P2", item["question_text"], item["options_text"], **demo) + build_answer_instruction(option_labels)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=700).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_code = parse_answer_from_text(gen_text, option_labels)
            pred_code_idx = item["code_to_index"].get(int(pred_code)) if pred_code is not None else None

            idx_to_label = dict(zip(item["ordinal_values"], (l.split(". ", 1)[1] for l in item["options_text"].splitlines())))
            rows.append({
                "respondent_id": int(row["respondent_id"]),
                "question_id": question_id,
                "question_text": item["question_text"],
                "condition": "P2",
                "model": f"trackB_{args.model.replace('/', '_')}_fold{args.fold}",
                "true_code": true_code,
                "true_label": idx_to_label.get(true_code),
                "true_code_idx": item["code_to_index"][true_code],
                "pred_code": int(pred_code) if pred_code is not None else None,
                "pred_label": idx_to_label.get(int(pred_code)) if pred_code is not None else None,
                "pred_code_idx": pred_code_idx,
                "pred_raw_text": gen_text,
                "refusal": pred_code is None,
            })
        logger.info(f"  ...respondent {row['respondent_id']} done ({len(rows)} predictions so far)")

    pred_df = pd.DataFrame(rows)
    out_path = Path(RESULTS_DIR) / "predictions" / f"trackB_{args.model.replace('/', '_')}_fold{args.fold}_P2.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path)

    n = len(pred_df)
    n_answered = pred_df["pred_code_idx"].notna().sum()
    acc = (pred_df["pred_code_idx"] == pred_df["true_code_idx"]).mean()
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY -- paste this back")
    logger.info(f"  model={args.model} fold={args.fold} n_items={len(selected_items)}")
    logger.info(f"  n_predictions={n} n_answered={n_answered} refusal_rate={1 - n_answered/n:.3f}")
    logger.info(f"  raw_accuracy={acc:.4f}")
    logger.info(f"  saved to {out_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
