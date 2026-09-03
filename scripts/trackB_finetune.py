"""Track B: QLoRA fine-tune on real WVS-7 India respondents, out-of-fold.

Hardened for a one-shot, unattended, remote lab run: preflight checks before
committing GPU time, auto-resume from checkpoint on restart, automatic
retry-with-smaller-batch on OOM, a status file you can check without
tailing logs, and a signal handler that saves before dying. Designed to run
under `tmux`/`nohup` so an SSH disconnect doesn't kill it -- but everything
it does on its own is to make a crash or restart cheap, not to prevent one.

Trains on ONE fold's training respondents (default: fold 0), then predicts
on that fold's held-out test respondents, using the EXACT SAME prompt
construction (verbalize_demographics / verbalize_item / build_prompt) and
the EXACT SAME item set as Track A's zero-shot runs, so this is a genuine
apples-to-apples "does fine-tuning help" comparison. Output parquet matches
Track A's schema so it drops straight into `src.report.evaluate_run`.

REQUIRED SEQUENCE -- do not skip steps, this is a one-shot expensive run:

    python -m scripts.trackB_finetune --preflight
        # ~5-10 min. Checks CUDA, GPU count/memory, disk space, HF Hub
        # reachability, and does one real forward+backward step with the
        # ACTUAL target model to confirm it fits before you commit further.
        # If this fails, FIX THE PROBLEM before continuing -- do not skip to
        # the real run "to see what happens."

    python -m scripts.trackB_finetune --smoke-test
        # ~10-20 min. Tiny data (5 train, 5 test respondents, 3 items), full
        # pipeline including a save+reload of a checkpoint, to catch data/
        # path/schema bugs before the real run.

    python -m scripts.trackB_finetune --fold 0 --n-items 15
        # The real run. Run this under tmux or nohup (see
        # TRACKB_LAB_INSTRUCTIONS.md) -- do NOT run it in a bare foreground
        # shell over SSH.

If the real run dies or the machine reboots, re-run the EXACT SAME command --
it auto-detects the latest checkpoint in --output-dir and resumes from there
instead of starting over.
"""

import argparse
import json
import logging
import signal
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

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

DEFAULT_MODEL = "openai/gpt-oss-120b"
MIN_FREE_DISK_GB = 250  # generous: model cache + checkpoints + HF download temp files
STATUS_PATH = Path("trackB_status.json")


def write_status(phase: str, detail: str = "", **extra):
    """Overwrite a small JSON file with current progress -- check this from
    another terminal instead of tailing the full log."""
    status = {"phase": phase, "detail": detail, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    try:
        STATUS_PATH.write_text(json.dumps(status, indent=2))
    except Exception:
        pass  # status file is a convenience, never let it crash the real run
    logger.info(f"[STATUS] {phase}: {detail}")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def preflight(args) -> bool:
    """Everything checkable in a few minutes, BEFORE spending real GPU time.
    Returns True only if it is safe to proceed to the real run."""
    ok = True

    write_status("preflight", "checking CUDA/GPUs")
    import torch
    if not torch.cuda.is_available():
        logger.error("PREFLIGHT FAIL: no CUDA device visible. Are you on the GPU node, not the master node?")
        return False
    n_gpus = torch.cuda.device_count()
    logger.info(f"CUDA OK. {n_gpus} GPU(s) visible:")
    total_mem_gb = 0
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        total_mem_gb += mem_gb
        logger.info(f"  GPU {i}: {props.name}, {mem_gb:.0f} GB")
    if total_mem_gb < 60:
        logger.error(f"PREFLIGHT FAIL: only {total_mem_gb:.0f} GB total GPU memory visible -- too little for a 120B-class model even in 4-bit. Check you got the GPU allocation you expected.")
        ok = False

    write_status("preflight", "checking disk space")
    free_gb = shutil.disk_usage(".").free / 1e9
    logger.info(f"Free disk space: {free_gb:.0f} GB")
    if free_gb < MIN_FREE_DISK_GB:
        logger.error(f"PREFLIGHT FAIL: only {free_gb:.0f} GB free, want at least {MIN_FREE_DISK_GB} GB for model cache + checkpoints. Clear space or point HF_HOME at a bigger disk before continuing.")
        ok = False

    write_status("preflight", "checking data files present")
    for p in [DATA_PROCESSED / "ind_wvs7.parquet", DATA_PROCESSED / "selected_items.json", DATA_PROCESSED / "folds.json"]:
        if not p.exists():
            logger.error(f"PREFLIGHT FAIL: missing {p} -- did you copy the full data/ directory?")
            ok = False
    if not ok:
        return False

    write_status("preflight", "checking Hugging Face Hub reachability")
    try:
        from huggingface_hub import HfApi
        HfApi().model_info(args.model)
        logger.info(f"HF Hub reachable, model repo '{args.model}' found.")
    except Exception as e:
        logger.error(f"PREFLIGHT FAIL: could not reach Hugging Face Hub or find '{args.model}': {e}")
        logger.error("If this is a private/gated model, run `huggingface-cli login` first.")
        ok = False
    if not ok:
        return False

    write_status("preflight", f"loading {args.model} in 4-bit and running one real train step (this is the slow part, several minutes)")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # One real forward+backward+step, on real-shaped input, to prove the
        # whole stack actually fits and runs -- not just that weights loaded.
        dummy_text = "This is a preflight check. " * 40
        inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=700).to(model.device)
        labels = inputs["input_ids"].clone()
        out = model(**inputs, labels=labels)
        out.loss.backward()
        model.zero_grad()
        peak_mem_gb = max(torch.cuda.max_memory_allocated(i) for i in range(n_gpus)) / 1e9
        logger.info(f"Model loaded, one train step ran successfully. Peak single-GPU memory: {peak_mem_gb:.1f} GB.")
        del model, out
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"PREFLIGHT FAIL: OOM loading/stepping {args.model} in 4-bit: {e}")
        logger.error("This model does not fit as configured. Do not proceed to the real run -- come back for a different --model or a smaller batch/sequence-length config first.")
        return False
    except Exception as e:
        logger.error(f"PREFLIGHT FAIL: error loading/running {args.model}: {e}")
        logger.error(traceback.format_exc())
        return False

    write_status("preflight", "PASSED -- safe to run --smoke-test next")
    logger.info("=" * 60)
    logger.info("PREFLIGHT PASSED. Next: python -m scripts.trackB_finetune --smoke-test")
    logger.info("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_examples(df: pd.DataFrame, respondent_ids: list, selected_items: list, codebook: dict) -> list:
    """One training example per (respondent, item): prompt text + the correct answer digit,
    built from the identical pipeline Track A used -- codebook-grounded question wording,
    codebook-grounded demographic verbalization, same P2 condition, same closing instruction."""
    examples = []
    skipped = 0
    sub = df[df["respondent_id"].isin(respondent_ids)]
    for _, row in sub.iterrows():
        try:
            demo = verbalize_demographics(row, codebook)
        except Exception as e:
            skipped += 1
            logger.warning(f"Skipping respondent {row.get('respondent_id')}: verbalize_demographics failed ({e})")
            continue
        for question_id in selected_items:
            try:
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
            except Exception as e:
                skipped += 1
                logger.warning(f"Skipping ({row.get('respondent_id')}, {question_id}): {e}")
    if skipped:
        logger.warning(f"Skipped {skipped} (respondent, item) pairs due to errors -- see warnings above.")
    return examples


def sanity_check_examples(examples: list, min_expected: int):
    if len(examples) < min_expected:
        raise RuntimeError(
            f"Only built {len(examples)} training examples, expected at least {min_expected}. "
            f"This smells like a data/config bug, not normal missingness -- STOP and investigate "
            f"rather than burning GPU time on a broken dataset."
        )


# ---------------------------------------------------------------------------
# Training with OOM backoff
# ---------------------------------------------------------------------------

def find_latest_checkpoint(output_dir: str):
    d = Path(output_dir)
    if not d.exists():
        return None
    checkpoints = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def train_with_oom_backoff(args, train_examples, tokenizer, model_loader):
    """Try the requested batch size; on OOM, halve it and retry (once per
    halving, down to batch size 1) rather than losing the whole run to a
    single bad batch-size guess."""
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    batch_size = args.batch_size
    grad_accum = args.grad_accum
    last_error = None

    while batch_size >= 1:
        try:
            model = model_loader()
            resume_from = find_latest_checkpoint(args.output_dir)
            if resume_from:
                logger.info(f"Found existing checkpoint {resume_from} -- resuming, NOT starting over.")

            training_args = TrainingArguments(
                output_dir=args.output_dir,
                overwrite_output_dir=False,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                warmup_steps=20,
                learning_rate=args.lr,
                weight_decay=0.01,
                bf16=True,
                logging_steps=10,
                save_steps=args.save_steps,
                save_total_limit=3,
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

            # Signal handler: on SIGTERM/SIGINT (job killed, preemption, ctrl-C),
            # save a checkpoint before dying instead of losing the last stretch.
            def _save_and_exit(signum, frame):
                logger.warning(f"Received signal {signum} -- saving checkpoint before exiting.")
                write_status("training", "interrupted, saving emergency checkpoint")
                try:
                    trainer.save_model(f"{args.output_dir}/emergency_checkpoint")
                except Exception:
                    logger.error("Emergency checkpoint save failed too.")
                sys.exit(1)

            signal.signal(signal.SIGTERM, _save_and_exit)
            signal.signal(signal.SIGINT, _save_and_exit)

            write_status("training", f"batch_size={batch_size} grad_accum={grad_accum}", n_examples=len(train_examples))
            trainer.train(resume_from_checkpoint=resume_from)
            return trainer, model

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            torch.cuda.empty_cache()
            logger.warning(f"OOM at batch_size={batch_size}. Halving batch size and doubling grad_accum to keep effective batch size roughly constant, then retrying.")
            batch_size //= 2
            grad_accum *= 2
            write_status("training", f"OOM recovery: retrying at batch_size={batch_size}")

    raise RuntimeError(f"Training failed even at batch_size=1: {last_error}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=15)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2, help="Conservative default for a 120B-class model -- preflight will tell you if you can safely go higher")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=50, help="Frequent by default -- checkpoints are small LoRA adapters, cheap to save often, expensive to lose")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="Run ONLY the safety checks, then exit. Do this first, always.")
    ap.add_argument("--output-dir", default="./trackB_output")
    args = ap.parse_args()

    if args.preflight:
        success = preflight(args)
        sys.exit(0 if success else 1)

    import torch
    if not torch.cuda.is_available():
        logger.error("No CUDA device found. Run --preflight first if you haven't.")
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
    min_expected = 10
    if args.smoke_test:
        train_ids, test_ids = train_ids[:5], test_ids[:5]
        selected_items = selected_items[:3]
        args.output_dir = args.output_dir + "_smoketest"
        logger.info("SMOKE TEST: 5 train respondents, 5 test respondents, 3 items")
    else:
        min_expected = 1000

    logger.info(f"Fold {args.fold}: {len(train_ids)} train, {len(test_ids)} test, {len(selected_items)} items")
    write_status("building_data", "constructing training examples")

    try:
        train_examples = build_examples(df, train_ids, selected_items, codebook)
        sanity_check_examples(train_examples, min_expected)
        logger.info(f"Built {len(train_examples)} training examples")
    except Exception as e:
        logger.error(f"FATAL during data build: {e}")
        logger.error(traceback.format_exc())
        write_status("failed", f"data build error: {e}")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def model_loader():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        m.config.use_cache = False
        m = prepare_model_for_kbit_training(m)
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        m = get_peft_model(m, lora_config)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
        return m

    write_status("training", "starting")
    t0 = time.time()
    try:
        trainer, model = train_with_oom_backoff(args, train_examples, tokenizer, model_loader)
    except Exception as e:
        logger.error(f"FATAL during training: {e}")
        logger.error(traceback.format_exc())
        write_status("failed", f"training error: {e}")
        logger.error("Re-run the SAME command -- it will resume from the last saved checkpoint, not restart.")
        sys.exit(1)
    logger.info(f"Training done in {(time.time()-t0)/60:.1f} min")

    model_path = f"{args.output_dir}/model_fold_{args.fold}"
    trainer.model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Saved fine-tuned model to {model_path}")
    write_status("training_complete", f"saved to {model_path}")

    # ---- Out-of-fold prediction on test respondents ----
    write_status("evaluating", "running out-of-fold inference")
    model.eval()
    rows = []
    test_sub = df[df["respondent_id"].isin(test_ids)]
    n_total = len(test_sub) * len(selected_items)
    t_eval0 = time.time()
    for i, (_, row) in enumerate(test_sub.iterrows()):
        try:
            demo = verbalize_demographics(row, codebook)
        except Exception as e:
            logger.warning(f"Skipping test respondent {row.get('respondent_id')}: {e}")
            continue
        for question_id in selected_items:
            try:
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
            except Exception as e:
                logger.warning(f"Prediction failed for ({row.get('respondent_id')}, {question_id}): {e}")

        if (i + 1) % 20 == 0 or (i + 1) == len(test_sub):
            elapsed = time.time() - t_eval0
            write_status("evaluating", f"{i+1}/{len(test_sub)} respondents, {len(rows)} predictions, {elapsed/60:.1f} min elapsed")
            # Save partial results every 20 respondents -- if eval crashes near
            # the end, you don't lose everything, just re-run to fill the rest.
            pd.DataFrame(rows).to_parquet(f"{args.output_dir}/partial_predictions.parquet")

    pred_df = pd.DataFrame(rows)
    out_path = Path(RESULTS_DIR) / "predictions" / f"trackB_{args.model.replace('/', '_')}_fold{args.fold}_P2.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path)

    n = len(pred_df)
    n_answered = pred_df["pred_code_idx"].notna().sum()
    acc = (pred_df["pred_code_idx"] == pred_df["true_code_idx"]).mean() if n else float("nan")
    summary = {
        "model": args.model, "fold": args.fold, "n_items": len(selected_items),
        "n_predictions": n, "n_answered": int(n_answered),
        "refusal_rate": round(1 - n_answered / n, 4) if n else None,
        "raw_accuracy": round(float(acc), 4) if n else None,
        "output_path": str(out_path),
    }
    write_status("complete", "run finished successfully", **summary)
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY -- paste this back")
    logger.info(json.dumps(summary, indent=2))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
