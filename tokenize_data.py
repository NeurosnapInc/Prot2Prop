"""
Pre-tokenize every task from the aggregated DuckDB into fixed train/validation/test
tensor splits for faster training.
"""

import random
import re

import duckdb
import torch
from transformers import T5Tokenizer

from config import (
  AGGREGATED_DB_PATH,
  MAX_LENGTH,
  MODEL_NAME,
  SPLIT_SEED,
  TEST_FRACTION,
  TOKENIZED_DATA_DIR,
  TRAIN_FRACTION,
  VAL_FRACTION,
)

TOKENIZE_BATCH_SIZE = 128


def _validate_split_fractions():
  total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
  if abs(total - 1.0) > 1e-8:
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def _preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  return "<AA2fold> " + " ".join(seq)


def _label_from_dtype(label: float, dtype: str):
  if dtype == "bool":
    return 1 if float(label) > 0.5 else 0
  if dtype == "int":
    return int(round(float(label)))
  return float(label)


print("Loading task data from DuckDB")
_validate_split_fractions()
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(AGGREGATED_DB_PATH)
try:
  task_rows = con.execute(
    """
    SELECT task_name, dtype, head_type, num_classes, loss
    FROM tasks
    ORDER BY task_name
    """
  ).fetchall()

  for task_name, dtype, head_type, num_classes, loss in task_rows:
    rows = con.execute(
      """
      SELECT sequence, label
      FROM samples
      WHERE task_name = ?
      """,
      [task_name],
    ).fetchall()

    meta = {
      "task_name": task_name,
      "dtype": dtype,
      "head_type": head_type,
      "num_classes": num_classes,
      "loss": loss,
    }

    normalized = []
    for sequence, label in rows:
      sequence = (sequence or "").strip()
      if not sequence:
        continue
      normalized.append({"sequence": sequence, "label": _label_from_dtype(label, dtype)})

    if not normalized:
      print(f"Task={task_name} skipped_empty=1")
      continue

    indices = list(range(len(normalized)))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(TRAIN_FRACTION * n_total)
    n_val = int(VAL_FRACTION * n_total)
    n_test = n_total - n_train - n_val

    splits = {
      "train": [normalized[i] for i in indices[:n_train]],
      "validation": [normalized[i] for i in indices[n_train:n_train + n_val]],
      "test": [normalized[i] for i in indices[n_train + n_val:n_train + n_val + n_test]],
    }

    if len(splits["train"]) == 0 or len(splits["validation"]) == 0:
      raise ValueError(f"Task '{task_name}' has an empty train/validation split; adjust dataset size or split fractions.")

    print(f"Task={task_name} dtype={dtype} head={head_type} loss={loss}")
    print(f"Rows: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")

    tokenized_splits = {}
    for split_name, split_rows in splits.items():
      input_id_batches = []
      attention_mask_batches = []
      label_batches = []

      for start in range(0, len(split_rows), TOKENIZE_BATCH_SIZE):
        batch = split_rows[start:start + TOKENIZE_BATCH_SIZE]
        sequences = [_preprocess_sequence(item["sequence"]) for item in batch]
        encoded = tokenizer(
          sequences,
          padding="max_length",
          truncation=True,
          max_length=MAX_LENGTH,
          return_tensors="pt",
        )

        input_id_batches.append(encoded["input_ids"])
        attention_mask_batches.append(encoded["attention_mask"])

        if dtype in ("bool", "int"):
          label_batches.append(torch.tensor([item["label"] for item in batch], dtype=torch.long))
        else:
          label_batches.append(torch.tensor([item["label"] for item in batch], dtype=torch.float).unsqueeze(-1))

      tokenized_splits[split_name] = {
        "input_ids": torch.cat(input_id_batches, dim=0),
        "attention_mask": torch.cat(attention_mask_batches, dim=0),
        "labels": torch.cat(label_batches, dim=0),
      }

    out_path = TOKENIZED_DATA_DIR / f"{task_name}_prostt5_tokens.pt"
    torch.save(
      {
        "meta": meta,
        "config": {
          "model_name": MODEL_NAME,
          "db_path": AGGREGATED_DB_PATH,
          "task_name": task_name,
          "split_seed": SPLIT_SEED,
          "train_fraction": TRAIN_FRACTION,
          "val_fraction": VAL_FRACTION,
          "test_fraction": TEST_FRACTION,
          "max_length": MAX_LENGTH,
        },
        "splits": tokenized_splits,
      },
      out_path,
    )
    print(f"Saved tokenized splits -> {out_path}")
finally:
  con.close()
