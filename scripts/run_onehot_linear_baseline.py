#!/usr/bin/env python3
"""Fit single-task one-hot linear baselines on the canonical sequence splits."""

import argparse
import hashlib
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
  accuracy_score,
  average_precision_score,
  balanced_accuracy_score,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  precision_score,
  recall_score,
  roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, label_binarize
from transformers import T5Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from config import AGGREGATED_DB_PATH, MAX_LENGTH, MODEL_NAME, SPLIT_SEED, TEST_FRACTION, TRAIN_CACHE_PATH, TRAIN_FRACTION, VAL_FRACTION  # noqa: E402


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
UNK_TOKEN = "X"
PAD_TOKEN = "_"
TOKENIZE_BATCH_SIZE = 128


def _format_float(value):
  if value is None:
    return "-"
  return f"{value:.4f}"


def _format_table(title, columns, rows):
  if not rows:
    return f"{title}\n(no rows)\n"

  widths = [len(col) for col in columns]
  for row in rows:
    for idx, cell in enumerate(row):
      widths[idx] = max(widths[idx], len(str(cell)))

  def render_row(row):
    return "  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

  divider = "  ".join("-" * width for width in widths)
  lines = [title, render_row(columns), divider]
  lines.extend(render_row(row) for row in rows)
  return "\n".join(lines) + "\n"


def _label_ratio_string(labels):
  if not labels:
    return "-"

  counts = Counter(labels)
  total = len(labels)
  return " ".join(f"{label}:{counts[label] / total:.3f}" for label in sorted(counts))


def _average_ranks(values):
  pairs = sorted(enumerate(values), key=lambda item: item[1])
  ranks = [0.0] * len(values)
  idx = 0
  while idx < len(pairs):
    end = idx + 1
    while end < len(pairs) and pairs[end][1] == pairs[idx][1]:
      end += 1
    avg_rank = (idx + 1 + end) / 2.0
    for pos in range(idx, end):
      ranks[pairs[pos][0]] = avg_rank
    idx = end
  return ranks


def _spearman_correlation(labels, preds):
  if len(labels) < 2:
    return None

  label_ranks = np.asarray(_average_ranks(labels), dtype=np.float64)
  pred_ranks = np.asarray(_average_ranks(preds), dtype=np.float64)
  label_centered = label_ranks - label_ranks.mean()
  pred_centered = pred_ranks - pred_ranks.mean()
  denominator = np.sqrt(np.dot(label_centered, label_centered) * np.dot(pred_centered, pred_centered))
  if denominator == 0.0:
    return None
  return float(np.dot(label_centered, pred_centered) / denominator)


def _classification_report(labels, preds, scores, dtype):
  average = "binary" if dtype == "bool" else "macro"
  report = {
    "acc": accuracy_score(labels, preds),
    "balanced_acc": balanced_accuracy_score(labels, preds),
    "precision": precision_score(labels, preds, average=average, zero_division=0),
    "recall": recall_score(labels, preds, average=average, zero_division=0),
    "f1": f1_score(labels, preds, average=average, zero_division=0),
    "label_ratio": _label_ratio_string(labels),
    "pred_ratio": _label_ratio_string(preds),
  }

  try:
    if dtype == "bool":
      positive_scores = scores
      report["auroc"] = roc_auc_score(labels, positive_scores)
      report["auprc"] = average_precision_score(labels, positive_scores)
    else:
      classes = sorted(set(labels))
      labels_binarized = label_binarize(labels, classes=classes)
      report["auroc"] = roc_auc_score(labels, scores, multi_class="ovr", average="macro")
      report["auprc"] = average_precision_score(labels_binarized, scores, average="macro")
  except ValueError:
    report["auroc"] = None
    report["auprc"] = None

  return report


def _regression_report(labels, preds):
  labels_arr = np.asarray(labels, dtype=np.float64)
  preds_arr = np.asarray(preds, dtype=np.float64)
  return {
    "label_mean": float(labels_arr.mean()),
    "label_std": float(labels_arr.std()),
    "pred_mean": float(preds_arr.mean()),
    "pred_std": float(preds_arr.std()),
    "mae": mean_absolute_error(labels_arr, preds_arr),
    "rmse": math.sqrt(mean_squared_error(labels_arr, preds_arr)),
    "spearman": _spearman_correlation(labels, preds),
  }


def _validate_split_fractions():
  total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
  if abs(total - 1.0) > 1e-8:
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def _split_identity_sequence(sequence: str) -> str:
  return (sequence or "").strip()


def _feature_sequence(sequence: str) -> str:
  return re.sub(r"[UZOB]", UNK_TOKEN, sequence.upper())


def _preprocess_sequence(sequence: str) -> str:
  return "<AA2fold> " + " ".join(_feature_sequence(sequence))


def _label_from_dtype(label: float, dtype: str):
  if dtype == "bool":
    return int(float(label) > 0.5)
  if dtype == "int":
    return int(round(float(label)))
  return float(label)


def _empty_label_row(num_tasks: int):
  return [0.0] * num_tasks


def _load_sequence_records(con: duckdb.DuckDBPyConnection):
  rows = con.execute(
    """
    SELECT sequence, task_name, label
    FROM samples
    ORDER BY sequence, task_name
    """
  ).fetchall()
  task_rows = con.execute(
    """
    SELECT task_name, dtype, head_type, num_classes, loss
    FROM tasks
    ORDER BY task_name
    """
  ).fetchall()
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

  sequence_records = {}
  for sequence, task_name, label in rows:
    split_identity = _split_identity_sequence(sequence)
    if not split_identity:
      continue
    record = sequence_records.setdefault(
      split_identity,
      {
        "sequence": split_identity,
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

  return records, task_order, task_metas


def _build_global_splits(con: duckdb.DuckDBPyConnection):
  _validate_split_fractions()
  records, task_order, task_metas = _load_sequence_records(con)

  indices = list(range(len(records)))
  rng = random.Random(SPLIT_SEED)
  rng.shuffle(indices)

  n_total = len(indices)
  n_train = int(TRAIN_FRACTION * n_total)
  n_val = int(VAL_FRACTION * n_total)

  return {
    "train": [records[i] for i in indices[:n_train]],
    "validation": [records[i] for i in indices[n_train:n_train + n_val]],
    "test": [records[i] for i in indices[n_train + n_val:]],
  }, task_order, task_metas


def _fingerprint_components(input_ids, raw_labels, label_mask):
  input_ids_arr = np.asarray(input_ids, dtype=np.int32)
  raw_labels_arr = np.asarray(raw_labels, dtype=np.float32)
  label_mask_arr = np.asarray(label_mask, dtype=np.bool_)
  digest = hashlib.blake2b(digest_size=16)
  digest.update(input_ids_arr.tobytes())
  digest.update(raw_labels_arr.tobytes())
  digest.update(label_mask_arr.tobytes())
  return digest.digest()


def _build_cache_fingerprint_counters(payload):
  counters = {}
  for split_name, split_payload in payload["splits"].items():
    counter = Counter()
    for idx in range(len(split_payload["input_ids"])):
      key = _fingerprint_components(
        split_payload["input_ids"][idx],
        split_payload["raw_labels"][idx].tolist(),
        split_payload["label_mask"][idx].tolist(),
      )
      counter[key] += 1
    counters[split_name] = counter
  return counters


def _build_cached_splits(records, task_order, cache_path):
  payload = torch.load(cache_path, map_location="cpu")
  cache_task_order = payload["task_order"]
  if cache_task_order != task_order:
    raise ValueError(
      f"Task order mismatch between DB and cache. db={task_order} cache={cache_task_order}"
    )

  tokenizer = T5Tokenizer.from_pretrained(payload["config"].get("model_name", MODEL_NAME), do_lower_case=False)
  max_length = payload["config"].get("max_length", MAX_LENGTH)
  counters = _build_cache_fingerprint_counters(payload)
  split_records = {split_name: [] for split_name in payload["splits"]}
  unmatched = 0

  for start in range(0, len(records), TOKENIZE_BATCH_SIZE):
    batch = records[start:start + TOKENIZE_BATCH_SIZE]
    encoded = tokenizer(
      [_preprocess_sequence(item["sequence"]) for item in batch],
      padding=False,
      truncation=True,
      max_length=max_length,
      return_attention_mask=False,
    )

    for input_ids, record in zip(encoded["input_ids"], batch):
      key = _fingerprint_components(input_ids, record["labels"], record["mask"])
      assigned = None
      for split_name in ("train", "validation", "test"):
        if counters[split_name].get(key, 0) > 0:
          counters[split_name][key] -= 1
          if counters[split_name][key] == 0:
            del counters[split_name][key]
          split_records[split_name].append(record)
          assigned = split_name
          break
      if assigned is None:
        unmatched += 1

  leftovers = {split_name: sum(counter.values()) for split_name, counter in counters.items()}
  if any(leftovers.values()):
    raise ValueError(
      f"Failed to reconstruct cached splits exactly from DuckDB. "
      f"unmatched_db_records={unmatched} leftover_cache_records={leftovers}"
    )

  return split_records, payload, unmatched


def _build_task_dataset(records, task_idx):
  sequences = []
  labels = []
  for record in records:
    if not record["mask"][task_idx]:
      continue
    label = record["labels"][task_idx]
    sequences.append(_feature_sequence(record["sequence"]))
    labels.append(label)
  return sequences, labels


def _infer_max_length(train_sequences, eval_sequences, requested_max_length):
  if requested_max_length is not None:
    return requested_max_length
  lengths = [len(sequence) for sequence in train_sequences]
  lengths.extend(len(sequence) for sequence in eval_sequences)
  if not lengths:
    raise ValueError("Cannot infer max length from an empty dataset.")
  return max(lengths)


def _sequence_matrix(sequences, max_length):
  matrix = []
  for sequence in sequences:
    truncated = sequence[:max_length]
    row = list(truncated)
    if len(row) < max_length:
      row.extend([PAD_TOKEN] * (max_length - len(row)))
    matrix.append(row)
  return np.asarray(matrix, dtype=object)


def _fit_encoder(train_sequences, eval_sequences, max_length):
  encoder_kwargs = {
    "categories": [list(AMINO_ACIDS) + [UNK_TOKEN, PAD_TOKEN]] * max_length,
    "handle_unknown": "ignore",
    "dtype": np.float32,
  }
  try:
    encoder = OneHotEncoder(sparse_output=True, **encoder_kwargs)
  except TypeError:
    encoder = OneHotEncoder(sparse=True, **encoder_kwargs)
  train_matrix = encoder.fit_transform(_sequence_matrix(train_sequences, max_length))
  eval_matrix = encoder.transform(_sequence_matrix(eval_sequences, max_length))
  return train_matrix, eval_matrix


def _evaluate_classification(task_name, dtype, y_train, y_eval, x_train, x_eval, max_iter, class_weight):
  model = LogisticRegression(
    solver="saga",
    max_iter=max_iter,
    class_weight=class_weight,
    random_state=SPLIT_SEED,
  )
  model.fit(x_train, y_train)
  pred_labels = model.predict(x_eval)
  if dtype == "bool":
    pred_scores = model.predict_proba(x_eval)[:, 1]
  else:
    pred_scores = model.predict_proba(x_eval)
  metrics = _classification_report(y_eval, pred_labels.tolist(), pred_scores, dtype)
  return {
    "task": task_name,
    "dtype": dtype,
    "n": len(y_eval),
    **metrics,
  }


def _evaluate_regression(task_name, y_train, y_eval, x_train, x_eval, alpha):
  model = Ridge(alpha=alpha)
  model.fit(x_train, y_train)
  preds = model.predict(x_eval)
  metrics = _regression_report(y_eval, preds.tolist())
  return {
    "task": task_name,
    "n": len(y_eval),
    **metrics,
  }


def parse_args():
  parser = argparse.ArgumentParser(description="Run single-task one-hot linear baselines on the canonical Prot2Prop splits.")
  parser.add_argument("--db-path", default=AGGREGATED_DB_PATH, help="Path to the aggregated DuckDB database.")
  parser.add_argument(
    "--cache",
    default=str(TRAIN_CACHE_PATH),
    help="Path to the tokenized multitask cache used by validate.py. Use this to reproduce exact persisted splits.",
  )
  parser.add_argument(
    "--split",
    default="validation",
    choices=["validation", "test"],
    help="Held-out split to evaluate after fitting on the train split.",
  )
  parser.add_argument("--task", action="append", help="Optional task name filter. May be passed multiple times.")
  parser.add_argument(
    "--max-length",
    type=int,
    default=None,
    help="Optional residue length cap. Defaults to the longest sequence observed for each task across train + eval.",
  )
  parser.add_argument(
    "--logreg-max-iter",
    type=int,
    default=1000,
    help="Maximum iterations for logistic regression.",
  )
  parser.add_argument(
    "--ridge-alpha",
    type=float,
    default=1.0,
    help="L2 regularization strength for regression.",
  )
  parser.add_argument(
    "--class-weight",
    choices=["none", "balanced"],
    default="balanced",
    help="Class weighting for classification tasks.",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  class_weight = None if args.class_weight == "none" else args.class_weight

  con = duckdb.connect(str(args.db_path), read_only=True)
  try:
    records, task_order, task_metas = _load_sequence_records(con)
  finally:
    con.close()

  split_records, payload, unmatched = _build_cached_splits(records, task_order, args.cache)
  train_records = split_records["train"]
  eval_records = split_records[args.split]
  requested_tasks = set(args.task or [])
  task_names = [task_name for task_name in task_order if not requested_tasks or task_name in requested_tasks]
  task_to_idx = {task_name: idx for idx, task_name in enumerate(task_order)}

  classification_rows = []
  regression_rows = []

  print(
    f"Cached sequence split: train={len(train_records)} {args.split}={len(eval_records)} "
    f"cache={args.cache} unmatched_db_records={unmatched}"
  )

  for task_name in task_names:
    if task_name not in task_to_idx:
      raise ValueError(f"Unknown task '{task_name}'. Available tasks: {', '.join(task_order)}")

    task_idx = task_to_idx[task_name]
    train_sequences, y_train = _build_task_dataset(train_records, task_idx)
    eval_sequences, y_eval = _build_task_dataset(eval_records, task_idx)
    if not train_sequences or not eval_sequences:
      raise ValueError(
        f"Task '{task_name}' has labels(train/{args.split})={len(train_sequences)}/{len(eval_sequences)} after cached split reconstruction."
      )

    max_length = _infer_max_length(train_sequences, eval_sequences, args.max_length)
    print(
      f"Task={task_name} dtype={task_metas[task_name]['dtype']} labels(train/{args.split})={len(train_sequences)}/{len(eval_sequences)} "
      f"max_length={max_length}"
    )
    x_train, x_eval = _fit_encoder(train_sequences, eval_sequences, max_length)

    if task_metas[task_name]["dtype"] in ("bool", "int"):
      classification_rows.append(
        _evaluate_classification(
          task_name,
          task_metas[task_name]["dtype"],
          y_train,
          y_eval,
          x_train,
          x_eval,
          args.logreg_max_iter,
          class_weight,
        )
      )
    else:
      regression_rows.append(
        _evaluate_regression(
          task_name,
          y_train,
          y_eval,
          x_train,
          x_eval,
          args.ridge_alpha,
        )
      )

  classification_table = _format_table(
    f"One-Hot Logistic Regression ({args.split})",
    ["task", "dtype", "n", "acc", "bal_acc", "precision", "recall", "f1", "auroc", "auprc", "label_ratio", "pred_ratio"],
    [
      [
        row["task"],
        row["dtype"],
        row["n"],
        _format_float(row["acc"]),
        _format_float(row["balanced_acc"]),
        _format_float(row["precision"]),
        _format_float(row["recall"]),
        _format_float(row["f1"]),
        _format_float(row["auroc"]),
        _format_float(row["auprc"]),
        row["label_ratio"],
        row["pred_ratio"],
      ]
      for row in classification_rows
    ],
  )
  regression_table = _format_table(
    f"One-Hot Ridge Regression ({args.split})",
    ["task", "n", "label_mean", "label_std", "pred_mean", "pred_std", "mae", "rmse", "spearman"],
    [
      [
        row["task"],
        row["n"],
        _format_float(row["label_mean"]),
        _format_float(row["label_std"]),
        _format_float(row["pred_mean"]),
        _format_float(row["pred_std"]),
        _format_float(row["mae"]),
        _format_float(row["rmse"]),
        _format_float(row["spearman"]),
      ]
      for row in regression_rows
    ],
  )
  print()
  print(classification_table, end="")
  print(regression_table, end="")


if __name__ == "__main__":
  main()
