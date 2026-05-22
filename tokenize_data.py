"""
Pre-tokenize the aggregated DuckDB into one multitask cache with global sequence
splits, per-task label masks, and train-split normalization stats.
"""

import json
import random
import re
from pathlib import Path

import duckdb
import torch
from transformers import T5Tokenizer

from config import (
  AGGREGATED_DB_PATH,
  EVOLUTIONARY_ALIGNMENT_TASK,
  EVOLUTIONARY_TARGETS_PATH,
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
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA)}


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


def _load_evolutionary_targets(path):
  if path is None or str(path).strip() == "":
    return {}

  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"Evolutionary targets file not found: {path}")

  targets = {}
  with path.open() as handle:
    for line_no, line in enumerate(handle, start=1):
      if not line.strip():
        continue

      row = json.loads(line)
      task_name = row.get("task_name")
      source = row.get("source")
      sequence = (row.get("sequence") or "").strip()
      alignment_nll = row.get("alignment_nll")
      alignment_mask = row.get("alignment_mask")

      if task_name != EVOLUTIONARY_ALIGNMENT_TASK:
        continue

      if not source or not sequence:
        raise ValueError(f"Evolutionary target line {line_no} missing source or sequence")

      if alignment_nll is None or alignment_mask is None:
        raise ValueError(f"Evolutionary target line {line_no} missing alignment_nll or alignment_mask")

      if len(alignment_nll) != len(sequence):
        raise ValueError(
          f"Evolutionary target line {line_no} has len(alignment_nll)={len(alignment_nll)} "
          f"but len(sequence)={len(sequence)}"
        )

      if len(alignment_mask) != len(sequence):
        raise ValueError(
          f"Evolutionary target line {line_no} has len(alignment_mask)={len(alignment_mask)} "
          f"but len(sequence)={len(sequence)}"
        )

      key = (task_name, source, sequence)
      targets[key] = {
        "alignment_nll": [float(x) for x in alignment_nll],
        "alignment_mask": [bool(x) for x in alignment_mask],
      }

  print(f"Loaded evolutionary targets: {len(targets)} from {path}")
  return targets


def _token_to_residue(token: str):
  # SentencePiece tokenizers often prefix regular tokens with ▁. Other tokenizers may
  # use Ġ. Strip these markers and keep only one-letter amino-acid tokens.
  token = token.replace("▁", "").replace("Ġ", "").strip()
  if len(token) == 1 and token in "ACDEFGHIKLMNPQRSTVWYX":
    return token
  return None


def _residue_token_positions(sequence: str, token_ids, tokenizer):
  cleaned_sequence = re.sub(r"[UZOB]", "X", sequence.upper())
  tokens = tokenizer.convert_ids_to_tokens(list(token_ids))

  positions = []
  residue_idx = 0

  for token_pos, token in enumerate(tokens):
    if residue_idx >= len(cleaned_sequence):
      break

    residue = _token_to_residue(token)
    if residue is None:
      continue

    # Only advance when the token matches the next residue. This avoids accidentally
    # treating the <AA2fold> control token or special tokens as sequence residues.
    if residue == cleaned_sequence[residue_idx]:
      positions.append(token_pos)
      residue_idx += 1

  return positions


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
  alignment_nll = []
  alignment_mask = []
  residue_token_ids = []

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

      token_alignment_nll = torch.zeros(len(ids), dtype=torch.float)
      token_alignment_mask = torch.zeros(len(ids), dtype=torch.bool)
      token_residue_ids = torch.full((len(ids),), -100, dtype=torch.long)

      raw_alignment_nll = item.get("alignment_nll")
      raw_alignment_mask = item.get("alignment_mask")
      if raw_alignment_nll is not None and raw_alignment_mask is not None:
        residue_positions = _residue_token_positions(item["sequence"], ids, tokenizer)
        usable = min(len(residue_positions), len(raw_alignment_nll), len(raw_alignment_mask))

        cleaned_sequence = re.sub(r"[UZOB]", "X", item["sequence"].upper())

        for residue_idx in range(usable):
          token_pos = residue_positions[residue_idx]
          residue = cleaned_sequence[residue_idx]
          if residue in AA_TO_IDX:
            token_residue_ids[token_pos] = AA_TO_IDX[residue]

          if raw_alignment_mask[residue_idx]:
            token_alignment_nll[token_pos] = float(raw_alignment_nll[residue_idx])
            token_alignment_mask[token_pos] = True

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
      alignment_nll.append(token_alignment_nll)
      alignment_mask.append(token_alignment_mask)
      residue_token_ids.append(token_residue_ids)

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
    "alignment_nll": alignment_nll,
    "alignment_mask": alignment_mask,
    "residue_token_ids": residue_token_ids,
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

  evolutionary_targets = _load_evolutionary_targets(EVOLUTIONARY_TARGETS_PATH)
  evolutionary_target_matches = 0

  sample_rows = con.execute(
    """
    SELECT sequence, source, task_name, label
    FROM samples
    ORDER BY sequence, source, task_name
    """
  ).fetchall()

  sequence_records = {}
  for sequence, source, task_name, label in sample_rows:
    sequence = (sequence or "").strip()
    if not sequence:
      continue

    record = sequence_records.setdefault(
      sequence,
      {
        "sequence": sequence,
        "labels": _empty_label_row(len(task_order)),
        "mask": [False] * len(task_order),
        "alignment_nll": None,
        "alignment_mask": None,
        "evo_source": None,
      },
    )

    if task_name == EVOLUTIONARY_ALIGNMENT_TASK and evolutionary_targets:
      evo_key = (task_name, source, sequence)
      evo_target = evolutionary_targets.get(evo_key)
      if evo_target is not None:
        if record["alignment_nll"] is None:
          evolutionary_target_matches += 1
        record["alignment_nll"] = evo_target["alignment_nll"]
        record["alignment_mask"] = evo_target["alignment_mask"]
        record["evo_source"] = source

    task_idx = task_to_idx[task_name]
    record["labels"][task_idx] = _label_from_dtype(label, task_metas[task_name]["dtype"])
    record["mask"][task_idx] = True

  records = list(sequence_records.values())
  if evolutionary_targets:
    print(f"Matched evolutionary targets for {evolutionary_target_matches} unique sequences")

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
          "cache_format": "multitask_sequence_masked_evo_v1",
          "evolutionary_targets_path": str(EVOLUTIONARY_TARGETS_PATH) if EVOLUTIONARY_TARGETS_PATH else None,
          "evolutionary_alignment_task": EVOLUTIONARY_ALIGNMENT_TASK,
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
