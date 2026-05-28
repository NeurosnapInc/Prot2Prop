"""
Validate an ensemble of multitask ProstT5 adapter checkpoints.

This script mirrors `validate.py`, but averages predictions from multiple saved
checkpoints instead of evaluating a single member. To keep GPU memory practical,
the frozen ProstT5 encoder is run once per batch and each ensemble member supplies
only its adapter, task adapters, pooling layer, and task heads on top of the
shared encoder tokens.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
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
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5EncoderModel

from calibration import format_posthoc_classification_rows, format_posthoc_regression_rows
from config import (
  ADAPTER_DIM,
  BATCH_SIZE,
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  EVAL_MAX_TOKENS_PER_BATCH,
  MODEL_NAME,
  REGRESSION_HEAD_HIDDEN,
  TASK_ADAPTER_DIM,
  TOKENIZED_DATA_DIR,
)
from model import Adapter, AttnPool, MultiTaskBatchSampler, MultiTaskSequenceDataset, TaskHead, collate_multitask_batch, output_dim_from_meta


DEFAULT_CACHE_PATH = TOKENIZED_DATA_DIR / "multitask_prostt5_tokens.pt"
DEFAULT_CHECKPOINT_GLOB = "prostt5_multitask_adapter_best_*.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"


class EnsembleMember(nn.Module):
  """Lightweight per-checkpoint adapter/pool/head stack on top of shared encoder tokens."""

  def __init__(
    self,
    checkpoint,
    task_order,
    task_output_dims,
    task_metas,
    embed_dim,
  ):
    super().__init__()
    checkpoint_config = checkpoint["config"]
    adapter_dim = checkpoint_config.get("adapter_dim", ADAPTER_DIM)
    task_adapter_dim = checkpoint_config.get("task_adapter_dim", 0)
    dropout = checkpoint_config.get("dropout", DROPOUT)

    classification_head_hidden = checkpoint_config.get("classification_head_hidden", 0)
    regression_head_hidden = checkpoint_config.get("regression_head_hidden", 0)
    if classification_head_hidden == 0 and regression_head_hidden == 0:
      classification_head_hidden = 0
      regression_head_hidden = 0
    else:
      classification_head_hidden = checkpoint_config.get("classification_head_hidden", CLASSIFICATION_HEAD_HIDDEN)
      regression_head_hidden = checkpoint_config.get("regression_head_hidden", REGRESSION_HEAD_HIDDEN)

    self.adapter = Adapter(embed_dim, adapter_dim=adapter_dim, dropout_prob=dropout)
    self.task_adapters = nn.ModuleDict()
    self.pool = AttnPool(embed_dim, hidden=checkpoint_config.get("attn_pool_hidden", 256), dropout=dropout)
    self.heads = nn.ModuleDict()

    for task_name in task_order:
      if task_adapter_dim and task_adapter_dim > 0:
        self.task_adapters[task_name] = Adapter(embed_dim, adapter_dim=task_adapter_dim, dropout_prob=dropout)
      else:
        self.task_adapters[task_name] = nn.Identity()

      meta = task_metas[task_name]
      hidden_dim = regression_head_hidden if meta["dtype"] == "float" else classification_head_hidden
      self.heads[task_name] = TaskHead(
        embed_dim,
        task_output_dims[task_name],
        hidden_dim=hidden_dim,
        dropout=dropout,
      )

    self.adapter.load_state_dict(checkpoint["adapter_state_dict"])
    for task_name, state_dict in checkpoint.get("task_adapter_state_dicts", {}).items():
      self.task_adapters[task_name].load_state_dict(state_dict)
    self.pool.load_state_dict(checkpoint["pool_state_dict"])
    for task_name, state_dict in checkpoint["head_state_dicts"].items():
      self.heads[task_name].load_state_dict(state_dict)

  def forward(self, shared_tokens, attention_mask):
    # Each member first applies its shared adapter, then its task-specific residual
    # adapters. This reproduces the exact per-checkpoint forward path without keeping
    # a separate copy of the frozen ProstT5 backbone in memory.
    adapted_tokens = shared_tokens + self.adapter(shared_tokens)
    outputs = {}
    for task_name, head in self.heads.items():
      task_tokens = adapted_tokens + self.task_adapters[task_name](adapted_tokens)
      pooled = self.pool(task_tokens, attention_mask)
      outputs[task_name] = head(pooled)
    return outputs


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
  parts = [f"{label}:{counts[label] / total:.3f}" for label in sorted(counts)]
  return " ".join(parts)


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
      report["auroc"] = roc_auc_score(labels, scores)
      report["auprc"] = average_precision_score(labels, scores)
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
  labels_tensor = torch.tensor(labels, dtype=torch.float)
  preds_tensor = torch.tensor(preds, dtype=torch.float)
  return {
    "label_mean": labels_tensor.mean().item(),
    "label_std": labels_tensor.std(unbiased=False).item(),
    "pred_mean": preds_tensor.mean().item(),
    "pred_std": preds_tensor.std(unbiased=False).item(),
    "mae": mean_absolute_error(labels, preds),
    "rmse": math.sqrt(mean_squared_error(labels, preds)),
    "spearman": _spearman_correlation(labels, preds),
  }


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

  label_ranks = torch.tensor(_average_ranks(labels), dtype=torch.float)
  pred_ranks = torch.tensor(_average_ranks(preds), dtype=torch.float)

  label_centered = label_ranks - label_ranks.mean()
  pred_centered = pred_ranks - pred_ranks.mean()
  denominator = torch.sqrt((label_centered.pow(2).sum()) * (pred_centered.pow(2).sum()))
  if denominator.item() == 0.0:
    return None

  return (label_centered * pred_centered).sum().div(denominator).item()


def _default_checkpoints():
  return sorted(str(path) for path in Path(".").glob(DEFAULT_CHECKPOINT_GLOB))


def _load_checkpoints(checkpoint_paths):
  checkpoints = []
  for checkpoint_path in checkpoint_paths:
    checkpoints.append(torch.load(checkpoint_path, map_location="cpu"))
  return checkpoints


def _validate_checkpoint_compatibility(checkpoints):
  # The ensemble can tolerate different adapter/head widths because each member owns
  # its own lightweight stack. It does require the same frozen backbone and same task
  # layout so predictions line up task-by-task.
  reference = checkpoints[0]["config"]
  required_keys = ["model_name", "embed_dim", "task_names", "task_output_dims"]
  for checkpoint_idx, checkpoint in enumerate(checkpoints[1:], start=2):
    current = checkpoint["config"]
    for key in required_keys:
      if current.get(key) != reference.get(key):
        raise ValueError(
          f"Checkpoint #{checkpoint_idx} is incompatible for ensembling: "
          f"config[{key!r}] differs from the first checkpoint."
        )


def parse_args():
  parser = argparse.ArgumentParser(description="Validate an ensemble of multitask ProstT5 adapter checkpoints.")
  parser.add_argument(
    "--checkpoints",
    nargs="+",
    help=f"Checkpoint paths to ensemble. Defaults to files matching {DEFAULT_CHECKPOINT_GLOB!r} in the current directory.",
  )
  parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Path to the tokenized multitask cache.")
  parser.add_argument(
    "--split",
    default="validation",
    choices=["train", "validation", "test"],
    help="Dataset split to evaluate.",
  )
  parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Maximum sequence count per evaluation batch.")
  return parser.parse_args()


def main():
  args = parse_args()
  checkpoint_paths = args.checkpoints or _default_checkpoints()
  if not checkpoint_paths:
    raise FileNotFoundError(f"No checkpoints provided and no files matched {DEFAULT_CHECKPOINT_GLOB!r}.")

  print("Loading checkpoints and tokenized cache")
  checkpoints = _load_checkpoints(checkpoint_paths)
  _validate_checkpoint_compatibility(checkpoints)
  payload = torch.load(args.cache, map_location="cpu")

  task_order = payload["task_order"]
  task_metas = payload["task_metas"]
  split_payload = payload["splits"][args.split]
  train_split = payload["splits"]["train"]
  pad_token_id = payload["config"]["pad_token_id"]
  regression_means = payload["normalization"]["train_mean"].to(DEVICE)
  regression_stds = payload["normalization"]["train_std"].to(DEVICE)

  dataset = MultiTaskSequenceDataset(split_payload)
  loader = DataLoader(
    dataset,
    batch_sampler=MultiTaskBatchSampler(
      dataset,
      args.batch_size,
      max_tokens_per_batch=EVAL_MAX_TOKENS_PER_BATCH,
    ),
    collate_fn=lambda batch: collate_multitask_batch(batch, pad_token_id),
    pin_memory=PIN_MEMORY,
  )

  task_output_dims = {}
  for task_idx, task_name in enumerate(task_order):
    meta = task_metas[task_name]
    train_mask = train_split["label_mask"][:, task_idx]
    train_labels = train_split["raw_labels"][:, task_idx]
    task_output_dims[task_name] = output_dim_from_meta(meta, train_labels, train_mask)

  model_name = checkpoints[0]["config"].get("model_name", MODEL_NAME)
  base_model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
  if DEVICE.type == "cuda":
    base_model.bfloat16()
  base_model.eval()

  embed_dim = checkpoints[0]["config"]["embed_dim"]
  members = []
  for checkpoint in checkpoints:
    member = EnsembleMember(checkpoint, task_order, task_output_dims, task_metas, embed_dim=embed_dim).to(DEVICE)
    member.eval()
    members.append(member)

  predictions = {
    task_name: {
      "labels": [],
      "preds": [],
      "scores": [],
    }
    for task_name in task_order
  }

  print(f"Running ensemble evaluation on split='{args.split}' with {len(checkpoint_paths)} checkpoints")
  for checkpoint_path in checkpoint_paths:
    print(f" - {checkpoint_path}")

  with torch.no_grad():
    for input_ids, attn_mask, raw_labels, normalized_labels, label_mask in tqdm(loader, desc="Ensemble Validate"):
      input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
      attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
      raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
      label_mask = label_mask.to(DEVICE, non_blocking=PIN_MEMORY)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        shared_tokens = base_model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        member_outputs = [member(shared_tokens, attn_mask) for member in members]

      for task_idx, task_name in enumerate(task_order):
        mask = label_mask[:, task_idx]
        if not mask.any():
          continue

        meta = task_metas[task_name]
        if meta["dtype"] == "float":
          preds_norm = torch.stack([outputs[task_name][mask].squeeze(-1).float() for outputs in member_outputs]).mean(dim=0)
          preds = preds_norm * regression_stds[task_idx] + regression_means[task_idx]
          labels = raw_labels[mask, task_idx].float()
          predictions[task_name]["preds"].extend(preds.cpu().tolist())
          predictions[task_name]["labels"].extend(labels.cpu().tolist())
        else:
          probs = torch.stack(
            [torch.softmax(outputs[task_name][mask].float(), dim=1) for outputs in member_outputs]
          ).mean(dim=0)
          preds = probs.argmax(dim=1)
          labels = raw_labels[mask, task_idx].long()
          predictions[task_name]["preds"].extend(preds.cpu().tolist())
          predictions[task_name]["labels"].extend(labels.cpu().tolist())
          if meta["dtype"] == "bool":
            predictions[task_name]["scores"].extend(probs[:, 1].cpu().tolist())
          else:
            predictions[task_name]["scores"].extend(probs.cpu().tolist())

  print()
  print(f"Dataset size ({args.split}): {len(dataset)} sequences")
  print(f"Checkpoints: {len(checkpoint_paths)}")
  print(f"Cache: {args.cache}")
  print()

  classification_rows = []
  regression_rows = []
  for task_name in sorted(task_order):
    labels = predictions[task_name]["labels"]
    preds = predictions[task_name]["preds"]
    if not labels:
      continue

    meta = task_metas[task_name]
    labeled_count = len(labels)
    if meta["dtype"] in ("bool", "int"):
      report = _classification_report(labels, preds, predictions[task_name]["scores"], meta["dtype"])
      classification_rows.append(
        [
          task_name,
          meta["dtype"],
          labeled_count,
          _format_float(report["acc"]),
          _format_float(report["balanced_acc"]),
          _format_float(report["precision"]),
          _format_float(report["recall"]),
          _format_float(report["f1"]),
          _format_float(report["auroc"]),
          _format_float(report["auprc"]),
          report["label_ratio"],
          report["pred_ratio"],
        ]
      )
    else:
      report = _regression_report(labels, preds)
      regression_rows.append(
        [
          task_name,
          labeled_count,
          _format_float(report["label_mean"]),
          _format_float(report["label_std"]),
          _format_float(report["pred_mean"]),
          _format_float(report["pred_std"]),
          _format_float(report["mae"]),
          _format_float(report["rmse"]),
          _format_float(report["spearman"]),
        ]
      )

  print(
    _format_table(
      "Classification Tasks",
      ["task", "dtype", "n", "acc", "bal_acc", "precision", "recall", "f1", "auroc", "auprc", "label_ratio", "pred_ratio"],
      classification_rows,
    )
  )
  print(
    _format_table(
      "Regression Tasks",
      ["task", "n", "label_mean", "label_std", "pred_mean", "pred_std", "mae", "rmse", "spearman"],
      regression_rows,
    )
  )

  posthoc_classification_rows = format_posthoc_classification_rows(predictions, task_metas)
  posthoc_regression_rows = format_posthoc_regression_rows(predictions, task_metas)
  print(
    _format_table(
      "Post-hoc Classification Threshold Tuning (fit on internal half, report on held-out half)",
      ["task", "cal_n", "rep_n", "thr", "acc", "bal_acc", "precision", "recall", "f1", "auroc", "auprc", "label_ratio", "pred_ratio"],
      posthoc_classification_rows,
    )
  )
  print(
    _format_table(
      "Post-hoc Regression Calibration (fit on internal half, report on held-out half)",
      ["task", "cal_n", "rep_n", "slope", "intercept", "pred_mean", "pred_std", "mae", "rmse", "spearman"],
      posthoc_regression_rows,
    )
  )


if __name__ == "__main__":
  main()
