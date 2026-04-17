"""
Pre-tokenize the aggregated DuckDB into one multitask cache with global sequence
splits, per-task label masks, and train-split normalization stats.
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
OUT_PATH = TOKENIZED_DATA_DIR / "multitask_prostt5_tokens.pt"


def _validate_split_fractions():
  total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
  if abs(total - 1.0) > 1e-8:
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def _preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  return "<AA2fold> " + " ".join(seq)


def _label_from_dtype(label: float, dtype: str):
  if dtype == "bool":
    return 1.0 if float(label) > 0.5 else 0.0
  if dtype == "int":
    return float(int(round(float(label))))
  return float(label)


def _empty_label_row(num_tasks: int):
  return [0.0] * num_tasks


def _compute_regression_stats(records, task_order, task_metas):
  means = torch.zeros(len(task_order), dtype=torch.float)
  stds = torch.ones(len(task_order), dtype=torch.float)

  for task_idx, task_name in enumerate(task_order):
    if task_metas[task_name]["dtype"] != "float":
      continue

    values = [
      float(record["labels"][task_idx])
      for record in records
      if record["mask"][task_idx]
    ]
    if not values:
      raise ValueError(f"Task '{task_name}' has no regression labels in the train split.")

    values_tensor = torch.tensor(values, dtype=torch.float)
    means[task_idx] = values_tensor.mean()
    std = values_tensor.std(unbiased=False)
    stds[task_idx] = std if std.item() > 0 else 1.0

  return means, stds


def _build_tokenized_split(records, tokenizer, task_order, task_metas, means, stds):
  input_ids = []
  raw_labels = []
  normalized_labels = []
  label_mask = []
  lengths = []

  for start in range(0, len(records), TOKENIZE_BATCH_SIZE):
    batch = records[start:start + TOKENIZE_BATCH_SIZE]
    sequences = [_preprocess_sequence(item["sequence"]) for item in batch]
    encoded = tokenizer(
      sequences,
      padding=False,
      truncation=True,
      max_length=MAX_LENGTH,
      return_attention_mask=False,
    )

    for ids, item in zip(encoded["input_ids"], batch):
      ids_tensor = torch.tensor(ids, dtype=torch.long)
      label_tensor = torch.tensor(item["labels"], dtype=torch.float)
      mask_tensor = torch.tensor(item["mask"], dtype=torch.bool)
      normalized_tensor = label_tensor.clone()

      for task_idx, task_name in enumerate(task_order):
        if not mask_tensor[task_idx]:
          continue
        if task_metas[task_name]["dtype"] == "float":
          normalized_tensor[task_idx] = (label_tensor[task_idx] - means[task_idx]) / stds[task_idx]

      input_ids.append(ids_tensor)
      raw_labels.append(label_tensor)
      normalized_labels.append(normalized_tensor)
      label_mask.append(mask_tensor)
      lengths.append(len(ids))

  if raw_labels:
    raw_labels_tensor = torch.stack(raw_labels)
    normalized_labels_tensor = torch.stack(normalized_labels)
    label_mask_tensor = torch.stack(label_mask)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
  else:
    num_tasks = len(task_order)
    raw_labels_tensor = torch.empty((0, num_tasks), dtype=torch.float)
    normalized_labels_tensor = torch.empty((0, num_tasks), dtype=torch.float)
    label_mask_tensor = torch.empty((0, num_tasks), dtype=torch.bool)
    lengths_tensor = torch.empty((0,), dtype=torch.long)

  return {
    "input_ids": input_ids,
    "raw_labels": raw_labels_tensor,
    "normalized_labels": normalized_labels_tensor,
    "label_mask": label_mask_tensor,
    "lengths": lengths_tensor,
  }


print("Loading multitask data from DuckDB")
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

  if not task_rows:
    raise ValueError(f"No tasks found in {AGGREGATED_DB_PATH}")

  task_order = [task_name for task_name, _, _, _, _ in task_rows]
  task_to_idx = {task_name: idx for idx, task_name in enumerate(task_order)}
  task_metas = {}
  for task_name, dtype, head_type, num_classes, loss in task_rows:
    task_metas[task_name] = {
      "task_name": task_name,
      "dtype": dtype,
      "head_type": head_type,
      "num_classes": num_classes,
      "loss": loss,
    }

  sample_rows = con.execute(
    """
    SELECT sequence, task_name, label
    FROM samples
    ORDER BY sequence, task_name
    """
  ).fetchall()

  sequence_records = {}
  for sequence, task_name, label in sample_rows:
    sequence = (sequence or "").strip()
    if not sequence:
      continue

    record = sequence_records.setdefault(
      sequence,
      {
        "sequence": sequence,
        "labels": _empty_label_row(len(task_order)),
        "mask": [False] * len(task_order),
      },
    )
    task_idx = task_to_idx[task_name]
    record["labels"][task_idx] = _label_from_dtype(label, task_metas[task_name]["dtype"])
    record["mask"][task_idx] = True

  records = list(sequence_records.values())
  if not records:
    raise ValueError(f"No valid sequences found in {AGGREGATED_DB_PATH}")

  indices = list(range(len(records)))
  rng = random.Random(SPLIT_SEED)
  rng.shuffle(indices)

  n_total = len(indices)
  n_train = int(TRAIN_FRACTION * n_total)
  n_val = int(VAL_FRACTION * n_total)
  n_test = n_total - n_train - n_val

  split_records = {
    "train": [records[i] for i in indices[:n_train]],
    "validation": [records[i] for i in indices[n_train:n_train + n_val]],
    "test": [records[i] for i in indices[n_train + n_val:n_train + n_val + n_test]],
  }

  if len(split_records["train"]) == 0 or len(split_records["validation"]) == 0:
    raise ValueError("Train/validation split is empty; adjust dataset size or split fractions.")

  train_means, train_stds = _compute_regression_stats(split_records["train"], task_order, task_metas)

  print(f"Unique sequences: train={len(split_records['train'])} val={len(split_records['validation'])} test={len(split_records['test'])}")
  for task_name in task_order:
    task_idx = task_to_idx[task_name]
    counts = {
      split_name: sum(1 for record in rows if record["mask"][task_idx])
      for split_name, rows in split_records.items()
    }
    if counts["train"] == 0 or counts["validation"] == 0:
      raise ValueError(
        f"Task '{task_name}' has labels(train/val/test)="
        f"{counts['train']}/{counts['validation']}/{counts['test']} after global sequence splitting. "
        "Adjust split fractions or task coverage."
      )
    stats_msg = ""
    if task_metas[task_name]["dtype"] == "float":
      stats_msg = f" mean={train_means[task_idx].item():.4f} std={train_stds[task_idx].item():.4f}"
    print(
      f"Task={task_name} dtype={task_metas[task_name]['dtype']} head={task_metas[task_name]['head_type']} "
      f"loss={task_metas[task_name]['loss']} labels(train/val/test)="
      f"{counts['train']}/{counts['validation']}/{counts['test']}{stats_msg}"
    )

  tokenized_splits = {}
  for split_name, rows in split_records.items():
    tokenized_splits[split_name] = _build_tokenized_split(
      rows,
      tokenizer,
      task_order,
      task_metas,
      train_means,
      train_stds,
    )

  torch.save(
    {
      "task_order": task_order,
      "task_metas": task_metas,
      "config": {
        "model_name": MODEL_NAME,
        "db_path": AGGREGATED_DB_PATH,
        "split_seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "max_length": MAX_LENGTH,
        "pad_token_id": tokenizer.pad_token_id,
        "cache_format": "multitask_sequence_masked_v1",
      },
      "normalization": {
        "train_mean": train_means,
        "train_std": train_stds,
      },
      "splits": tokenized_splits,
    },
    OUT_PATH,
  )
  print(f"Saved multitask tokenized splits -> {OUT_PATH}")
finally:
  con.close()
