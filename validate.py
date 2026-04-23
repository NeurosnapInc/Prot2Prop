"""
Validate a trained multi-task ProstT5 adapter checkpoint on a cached split and
print tabular metrics for classification and regression tasks.
"""

import argparse
import math
from collections import Counter

import torch
import torch.nn as nn
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  precision_score,
  recall_score,
)
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import T5EncoderModel

from config import MODEL_NAME, TOKENIZED_DATA_DIR


DEFAULT_CHECKPOINT_PATH = "./prostt5_multitask_adapter_best.pt"
DEFAULT_CACHE_PATH = TOKENIZED_DATA_DIR / "multitask_prostt5_tokens.pt"
BATCH_SIZE = 32
ADAPTER_DIM = 64
DROPOUT = 0.1
ATTN_POOL_HIDDEN = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"


class MultiTaskSequenceDataset(Dataset):
  def __init__(self, split_payload):
    self.samples = []
    for idx, length in enumerate(split_payload["lengths"]):
      self.samples.append(
        {
          "input_ids": split_payload["input_ids"][idx],
          "raw_labels": split_payload["raw_labels"][idx],
          "normalized_labels": split_payload["normalized_labels"][idx],
          "label_mask": split_payload["label_mask"][idx],
          "length": int(length),
        }
      )

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


class MultiTaskBatchSampler(Sampler):
  def __init__(self, dataset, batch_size):
    self.dataset = dataset
    self.batch_size = batch_size

  def __iter__(self):
    indices = list(range(len(self.dataset)))
    indices.sort(key=lambda idx: self.dataset.samples[idx]["length"])
    batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
    return iter(batches)

  def __len__(self):
    return math.ceil(len(self.dataset) / self.batch_size)


class Adapter(nn.Module):
  def __init__(self, input_dim, adapter_dim=ADAPTER_DIM, dropout_prob=DROPOUT):
    super().__init__()
    self.norm = nn.LayerNorm(input_dim)
    self.down_project = nn.Linear(input_dim, adapter_dim)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(adapter_dim, input_dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.scale = nn.Parameter(torch.tensor(1e-3))
    nn.init.normal_(self.down_project.weight, std=1e-3)
    nn.init.normal_(self.up_project.weight, std=1e-3)
    nn.init.zeros_(self.up_project.bias)

  def forward(self, x):
    x_norm = self.norm(x)
    down = self.down_project(x_norm)
    activated = self.activation(down)
    up = self.up_project(activated)
    dropped = self.dropout(up)
    return self.scale * dropped


class AttnPool(nn.Module):
  def __init__(self, d_model, hidden=ATTN_POOL_HIDDEN, dropout=DROPOUT):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(d_model, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.context = nn.Linear(hidden, 1, bias=False)

  def forward(self, x, mask):
    h = self.proj(x)
    scores = self.context(h).squeeze(-1)
    scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=1)
    return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


class MultiTaskAdapterModel(nn.Module):
  def __init__(self, base_model, task_order, task_output_dims, embed_dim, adapter_dim=ADAPTER_DIM, dropout=DROPOUT):
    super().__init__()
    self.base = base_model
    self.adapter = Adapter(embed_dim, adapter_dim, dropout_prob=dropout)
    self.pool = AttnPool(embed_dim, hidden=ATTN_POOL_HIDDEN, dropout=dropout)
    self.heads = nn.ModuleDict()

    for task_name in task_order:
      output_dim = task_output_dims[task_name]
      self.heads[task_name] = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, output_dim),
      )

  def encode(self, input_ids, attention_mask):
    if input_ids.dtype != torch.long:
      input_ids = input_ids.long()
    if attention_mask.dtype not in (torch.long, torch.int64, torch.bool):
      attention_mask = attention_mask.long()

    out = self.base(input_ids=input_ids, attention_mask=attention_mask)
    token_repr = out.last_hidden_state
    adapted = token_repr + self.adapter(token_repr)
    return self.pool(adapted, attention_mask)

  def forward(self, input_ids, attention_mask):
    pooled = self.encode(input_ids, attention_mask)
    return {task_name: head(pooled) for task_name, head in self.heads.items()}


def _collate_batch(batch, pad_token_id):
  input_ids = [sample["input_ids"] for sample in batch]
  padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
  attention_mask = padded_ids.ne(pad_token_id).long()
  raw_labels = torch.stack([sample["raw_labels"] for sample in batch])
  normalized_labels = torch.stack([sample["normalized_labels"] for sample in batch])
  label_mask = torch.stack([sample["label_mask"] for sample in batch])
  return padded_ids, attention_mask, raw_labels, normalized_labels, label_mask


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


def _output_dim_from_meta(meta, labels, mask):
  if meta["dtype"] == "float":
    return 1

  if meta["num_classes"] is not None:
    return int(meta["num_classes"])

  observed = labels[mask]
  if observed.numel() == 0:
    raise ValueError(f"Task '{meta['task_name']}' has no observed labels in train split.")
  return int(observed.max().item()) + 1


def _label_ratio_string(labels):
  if not labels:
    return "-"

  counts = Counter(labels)
  total = len(labels)
  parts = [f"{label}:{counts[label] / total:.3f}" for label in sorted(counts)]
  return " ".join(parts)


def _classification_report(labels, preds, dtype):
  average = "binary" if dtype == "bool" else "macro"
  return {
    "acc": accuracy_score(labels, preds),
    "balanced_acc": balanced_accuracy_score(labels, preds),
    "precision": precision_score(labels, preds, average=average, zero_division=0),
    "recall": recall_score(labels, preds, average=average, zero_division=0),
    "f1": f1_score(labels, preds, average=average, zero_division=0),
    "label_ratio": _label_ratio_string(labels),
    "pred_ratio": _label_ratio_string(preds),
  }


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
    batch_sampler=MultiTaskBatchSampler(dataset, args.batch_size),
    collate_fn=lambda batch: _collate_batch(batch, pad_token_id),
    pin_memory=PIN_MEMORY,
  )

  task_output_dims = {}
  for task_idx, task_name in enumerate(task_order):
    meta = task_metas[task_name]
    train_mask = train_split["label_mask"][:, task_idx]
    train_labels = train_split["raw_labels"][:, task_idx]
    task_output_dims[task_name] = _output_dim_from_meta(meta, train_labels, train_mask)

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
          preds = outputs[task_name][mask].argmax(dim=1)
          labels = raw_labels[mask, task_idx].long()
          predictions[task_name]["preds"].extend(preds.cpu().tolist())
          predictions[task_name]["labels"].extend(labels.cpu().tolist())

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
      report = _classification_report(labels, preds, meta["dtype"])
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
      ["task", "dtype", "n", "acc", "bal_acc", "precision", "recall", "f1", "label_ratio", "pred_ratio"],
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
