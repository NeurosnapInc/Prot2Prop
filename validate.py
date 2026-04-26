"""
Validate a trained multi-task ProstT5 adapter checkpoint on a cached split and
print tabular metrics for classification and regression tasks.
"""

import argparse
import math
from collections import Counter

import torch
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

from config import ADAPTER_DIM, BATCH_SIZE, DROPOUT, EVAL_MAX_TOKENS_PER_BATCH, MODEL_NAME, TOKENIZED_DATA_DIR
from model import (
  MultiTaskAdapterModel,
  MultiTaskBatchSampler,
  MultiTaskSequenceDataset,
  collate_multitask_batch,
  output_dim_from_meta,
)


DEFAULT_CHECKPOINT_PATH = "./prostt5_multitask_adapter_best.pt"
DEFAULT_CACHE_PATH = TOKENIZED_DATA_DIR / "multitask_prostt5_tokens.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"


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


def parse_args():
  parser = argparse.ArgumentParser(description="Validate a trained multitask ProstT5 adapter checkpoint.")
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to the saved adapter checkpoint.")
  parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="Path to the tokenized multitask cache.")
  parser.add_argument(
    "--split",
    default="validation",
    choices=["train", "validation", "test"],
    help="Dataset split to evaluate.",
  )
  parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation.")
  return parser.parse_args()


def main():
  args = parse_args()

  print("Loading checkpoint and tokenized cache")
  checkpoint = torch.load(args.checkpoint, map_location="cpu")
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

  model_name = checkpoint["config"].get("model_name", MODEL_NAME)
  base_model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
  if DEVICE.type == "cuda":
    base_model.bfloat16()

  embed_dim = checkpoint["config"]["embed_dim"]
  model = MultiTaskAdapterModel(
    base_model,
    task_order,
    task_output_dims,
    embed_dim=embed_dim,
    adapter_dim=checkpoint["config"].get("adapter_dim", ADAPTER_DIM),
    dropout=checkpoint["config"].get("dropout", DROPOUT),
  ).to(DEVICE)

  model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
  model.pool.load_state_dict(checkpoint["pool_state_dict"])
  for task_name, state_dict in checkpoint["head_state_dicts"].items():
    model.heads[task_name].load_state_dict(state_dict)
  model.eval()

  predictions = {
    task_name: {
      "labels": [],
      "preds": [],
      "scores": [],
    }
    for task_name in task_order
  }

  print(f"Running evaluation on split='{args.split}'")
  with torch.no_grad():
    for input_ids, attn_mask, raw_labels, normalized_labels, label_mask in tqdm(loader, desc="Validate"):
      input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
      attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
      raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
      normalized_labels = normalized_labels.to(DEVICE, non_blocking=PIN_MEMORY)
      label_mask = label_mask.to(DEVICE, non_blocking=PIN_MEMORY)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(input_ids, attn_mask)

      for task_idx, task_name in enumerate(task_order):
        mask = label_mask[:, task_idx]
        if not mask.any():
          continue

        meta = task_metas[task_name]
        if meta["dtype"] == "float":
          preds_norm = outputs[task_name][mask].squeeze(-1).float()
          preds = preds_norm * regression_stds[task_idx] + regression_means[task_idx]
          labels = raw_labels[mask, task_idx].float()
          predictions[task_name]["preds"].extend(preds.cpu().tolist())
          predictions[task_name]["labels"].extend(labels.cpu().tolist())
        else:
          logits = outputs[task_name][mask].float()
          probs = torch.softmax(logits, dim=1)
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
  print(f"Checkpoint: {args.checkpoint}")
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


if __name__ == "__main__":
  main()
