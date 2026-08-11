"""Fine-tune a lightweight FLAN-T5 to generate eligibility criteria (or conditions)
from article text. Trains ONLY on provided (article -> gold) pairs. GPU for dev-time
training; the resulting model runs on CPU at inference (laptop-compliant).

Usage: python src/train_gen.py --data data/gen_elig.jsonl --out models/gen_elig --model google/flan-t5-base --epochs 12
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]


class JsonlDS(Dataset):
    def __init__(self, path, tok, max_src, max_tgt):
        self.rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok, self.max_src, self.max_tgt = tok, max_src, max_tgt

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        m = self.tok(r["input"], truncation=True, max_length=self.max_src)
        with self.tok.as_target_tokenizer():
            lab = self.tok(r["target"], truncation=True, max_length=self.max_tgt)
        m["labels"] = lab["input_ids"]
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "gen_elig.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "models" / "gen_elig"))
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_src", type=int, default=640)
    ap.add_argument("--max_tgt", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    ds = JsonlDS(args.data, tok, args.max_src, args.max_tgt)
    print(f"training {args.model} on {len(ds)} pairs -> {args.out}")

    # FLAN-T5 is numerically unstable in fp16 (NaN loss); use bf16 when available.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=args.out + "_ckpt", num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, learning_rate=args.lr,
        bf16=use_bf16, fp16=False, logging_steps=25, save_strategy="no",
        report_to="none", warmup_ratio=0.05, weight_decay=0.01)
    collator = DataCollatorForSeq2Seq(tok, model=model)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
