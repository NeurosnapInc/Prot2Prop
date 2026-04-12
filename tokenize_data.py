"""
Pre-tokenize one task from the aggregated DuckDB into fixed train/validation/test
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
  TASK_NAME,
  TEST_FRACTION,
  TOKENIZED_DATA_PATH,
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
con = duckdb.connect(AGGREGATED_DB_PATH)
try:
  task_row = con.execute(
    """
    SELECT task_name, dtype, head_type, num_classes, loss
    FROM tasks
    WHERE task_name = ?
    """,
    [TASK_NAME],
  ).fetchone()
  if task_row is None:
    raise ValueError(f"Task '{TASK_NAME}' not found in tasks table at {AGGREGATED_DB_PATH}")

  rows = con.execute(
    """
    SELECT sequence, label
    FROM samples
    WHERE task_name = ?
    """,
    [TASK_NAME],
  ).fetchall()
finally:
  con.close()

meta = {
  "task_name": task_row[0],
  "dtype": task_row[1],
  "head_type": task_row[2],
  "num_classes": task_row[3],
  "loss": task_row[4],
}

normalized = []
for sequence, label in rows:
  sequence = (sequence or "").strip()
  if not sequence:
    continue
  normalized.append({"sequence": sequence, "label": _label_from_dtype(label, meta["dtype"])})

if not normalized:
  raise ValueError(f"No usable samples found for task '{TASK_NAME}' in {AGGREGATED_DB_PATH}")

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
  raise ValueError("Train/validation split is empty; adjust dataset size or split fractions.")

print(f"Task={meta['task_name']} dtype={meta['dtype']} head={meta['head_type']} loss={meta['loss']}")
print(f"Rows: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
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

    if meta["dtype"] in ("bool", "int"):
      label_batches.append(torch.tensor([item["label"] for item in batch], dtype=torch.long))
    else:
      label_batches.append(torch.tensor([item["label"] for item in batch], dtype=torch.float).unsqueeze(-1))

  tokenized_splits[split_name] = {
    "input_ids": torch.cat(input_id_batches, dim=0),
    "attention_mask": torch.cat(attention_mask_batches, dim=0),
    "labels": torch.cat(label_batches, dim=0),
  }

TOKENIZED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(
  {
    "meta": meta,
    "config": {
      "model_name": MODEL_NAME,
      "db_path": AGGREGATED_DB_PATH,
      "task_name": TASK_NAME,
      "split_seed": SPLIT_SEED,
      "train_fraction": TRAIN_FRACTION,
      "val_fraction": VAL_FRACTION,
      "test_fraction": TEST_FRACTION,
      "max_length": MAX_LENGTH,
    },
    "splits": tokenized_splits,
  },
  TOKENIZED_DATA_PATH,
)
print(f"Saved tokenized splits -> {TOKENIZED_DATA_PATH}")
