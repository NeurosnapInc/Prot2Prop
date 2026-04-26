"""
Multi-task training script for one shared ProstT5 adapter with task-specific heads,
backed by one sequence-level tokenized cache with masked labels.
"""

import math
from collections import Counter
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5EncoderModel, get_linear_schedule_with_warmup

from config import (
  ADAPTER_DIM,
  ATTN_POOL_HIDDEN,
  BATCH_SAMPLER_SEED,
  BATCH_SIZE,
  DROPOUT,
  EVAL_MAX_TOKENS_PER_BATCH,
  EPOCHS,
  LR,
  MODEL_NAME,
  PATIENCE,
  TRAIN_CACHE_PATH,
  WARMUP_RATIO,
  WEIGHT_DECAY,
)
from model import (
  MultiTaskAdapterModel,
  MultiTaskBatchSampler,
  MultiTaskSequenceDataset,
  collate_multitask_batch,
  output_dim_from_meta,
  unwrap_model,
)


### Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
COMPILE_MODEL = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
USE_FUSED_ADAMW = DEVICE.type == "cuda"


def _build_classification_loss(meta: Dict[str, str], labels: torch.Tensor, mask: torch.Tensor):
  observed = labels[mask].long()
  if meta["dtype"] == "bool" and (meta["num_classes"] in (None, 2)):
    counts = Counter(int(x) for x in observed.tolist())
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    total = n0 + n1
    w0 = total / (2.0 * max(1, n0))
    w1 = total / (2.0 * max(1, n1))
    weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
    return nn.CrossEntropyLoss(weight=weights)
  return nn.CrossEntropyLoss()


def _metric_from_preds(labels, preds, dtype: str) -> Tuple[str, float, Dict[str, float]]:
  if dtype in ("bool", "int"):
    acc = accuracy_score(labels, preds)
    if dtype == "bool":
      f1 = f1_score(labels, preds, zero_division=0)
    else:
      f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return "f1", f1, {"acc": acc, "f1": f1}

  mae = mean_absolute_error(labels, preds)
  rmse = math.sqrt(mean_squared_error(labels, preds))
  return "mae", mae, {"mae": mae, "rmse": rmse}


def _compute_sample_weights(split_payload, task_order):
  label_mask = split_payload["label_mask"]
  task_counts = label_mask.sum(dim=0).float()
  inv_task_counts = torch.zeros_like(task_counts)
  nonzero = task_counts > 0
  inv_task_counts[nonzero] = 1.0 / task_counts[nonzero]

  sample_weights = []
  for row_mask in label_mask:
    present = row_mask.nonzero(as_tuple=False).view(-1)
    if present.numel() == 0:
      sample_weights.append(1.0)
      continue
    sample_weights.append(float(inv_task_counts[present].mean().item()))

  weights = torch.tensor(sample_weights, dtype=torch.double)
  weights /= weights.sum()

  task_label_counts = {
    task_name: int(task_counts[idx].item())
    for idx, task_name in enumerate(task_order)
  }
  return weights, task_label_counts


def _compute_multitask_loss(outputs, raw_labels, normalized_labels, label_mask, task_order, task_metas, criteria):
  task_losses = []

  for task_idx, task_name in enumerate(task_order):
    mask = label_mask[:, task_idx]
    if not mask.any():
      continue

    preds = outputs[task_name][mask]
    meta = task_metas[task_name]
    if meta["dtype"] == "float":
      targets = normalized_labels[mask, task_idx]
      if (meta["loss"] or "").lower() in ("mae", "l1"):
        task_loss = F.l1_loss(preds.squeeze(-1), targets)
      else:
        task_loss = F.mse_loss(preds.squeeze(-1), targets)
    else:
      targets = raw_labels[mask, task_idx].long()
      task_loss = criteria[task_name](preds, targets)

    task_losses.append(task_loss)

  if not task_losses:
    raise ValueError("Encountered a batch with no observed task labels.")

  return torch.stack(task_losses).mean()


print("Loading multitask tokenized cache")
if not TRAIN_CACHE_PATH.exists():
  raise FileNotFoundError(f"Missing multitask tokenized cache at {TRAIN_CACHE_PATH}. Run tokenize_data.py first.")

payload = torch.load(TRAIN_CACHE_PATH, map_location="cpu")
task_order = payload["task_order"]
task_metas = payload["task_metas"]
train_split = payload["splits"]["train"]
val_split = payload["splits"]["validation"]
pad_token_id = payload["config"]["pad_token_id"]
normalization = payload["normalization"]
regression_means = normalization["train_mean"]
regression_stds = normalization["train_std"]

train_ds = MultiTaskSequenceDataset(train_split)
val_ds = MultiTaskSequenceDataset(val_split)
train_sample_weights, train_label_counts = _compute_sample_weights(train_split, task_order)

print(f"Loaded multitask cache from {TRAIN_CACHE_PATH}")
print(f"Sequences: train={len(train_ds)} val={len(val_ds)}")
for task_idx, task_name in enumerate(task_order):
  meta = task_metas[task_name]
  train_count = train_label_counts[task_name]
  val_count = int(val_split["label_mask"][:, task_idx].sum().item())
  if meta["dtype"] == "float":
    stats_msg = f" mean={regression_means[task_idx].item():.4f} std={regression_stds[task_idx].item():.4f}"
  else:
    stats_msg = ""
  print(
    f"Task={task_name} dtype={meta['dtype']} head={meta['head_type']} loss={meta['loss']} "
    f"labels(train/val)={train_count}/{val_count}{stats_msg}"
  )

base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == "cuda":
  base_model.bfloat16()

train_loader = DataLoader(
  train_ds,
  batch_sampler=MultiTaskBatchSampler(
    train_ds,
    BATCH_SIZE,
    shuffle=True,
    seed=BATCH_SAMPLER_SEED,
    sample_weights=train_sample_weights,
  ),
  collate_fn=lambda batch: collate_multitask_batch(batch, pad_token_id),
  pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
  val_ds,
  batch_sampler=MultiTaskBatchSampler(
    val_ds,
    BATCH_SIZE,
    shuffle=False,
    seed=BATCH_SAMPLER_SEED,
    max_tokens_per_batch=EVAL_MAX_TOKENS_PER_BATCH,
  ),
  collate_fn=lambda batch: collate_multitask_batch(batch, pad_token_id),
  pin_memory=PIN_MEMORY,
)

print("Initializing model")
task_output_dims = {}
criteria = {}
for task_idx, task_name in enumerate(task_order):
  meta = task_metas[task_name]
  train_mask = train_split["label_mask"][:, task_idx]
  train_labels = train_split["raw_labels"][:, task_idx]
  task_output_dims[task_name] = output_dim_from_meta(meta, train_labels, train_mask)
  if meta["dtype"] != "float":
    criteria[task_name] = _build_classification_loss(meta, train_labels, train_mask)

embed_dim = base_model.config.d_model
model = MultiTaskAdapterModel(
  base_model,
  task_order,
  task_output_dims,
  embed_dim=embed_dim,
  adapter_dim=ADAPTER_DIM,
  dropout=DROPOUT,
).to(DEVICE)

if COMPILE_MODEL and hasattr(torch, "compile"):
  print("Compiling model")
  try:
    model = torch.compile(model)
  except Exception as exc:
    print(f"torch.compile unavailable, continuing without compile: {exc}")

model_ref = unwrap_model(model)
optimizer = torch.optim.AdamW(
  [{"params": model_ref.adapter.parameters()}, {"params": model_ref.pool.parameters()}, {"params": model_ref.heads.parameters()}],
  lr=LR,
  weight_decay=WEIGHT_DECAY,
  fused=USE_FUSED_ADAMW,
)
trainable_params = list(model_ref.adapter.parameters()) + list(model_ref.pool.parameters()) + list(model_ref.heads.parameters())

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_metric = -float("inf")
stale = 0
best_state = None

regression_means_device = regression_means.to(DEVICE)
regression_stds_device = regression_stds.to(DEVICE)

for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0

  for input_ids, attn_mask, raw_labels, normalized_labels, label_mask in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
    input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
    attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
    raw_labels = raw_labels.to(DEVICE, non_blocking=PIN_MEMORY)
    normalized_labels = normalized_labels.to(DEVICE, non_blocking=PIN_MEMORY)
    label_mask = label_mask.to(DEVICE, non_blocking=PIN_MEMORY)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
      outputs = model(input_ids, attn_mask)
      loss = _compute_multitask_loss(outputs, raw_labels, normalized_labels, label_mask, task_order, task_metas, criteria)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  model.eval()
  val_predictions = {
    task_name: {
      "preds": [],
      "labels": [],
      "normalized_preds": [],
      "normalized_labels": [],
    }
    for task_name in task_order
  }
  with torch.no_grad():
    for input_ids, attn_mask, raw_labels, normalized_labels, label_mask in val_loader:
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
          labels_norm = normalized_labels[mask, task_idx].float()
          preds = preds_norm * regression_stds_device[task_idx] + regression_means_device[task_idx]
          labels = raw_labels[mask, task_idx].float()
          val_predictions[task_name]["preds"].extend(preds.cpu().numpy().tolist())
          val_predictions[task_name]["labels"].extend(labels.cpu().numpy().tolist())
          val_predictions[task_name]["normalized_preds"].extend(preds_norm.cpu().numpy().tolist())
          val_predictions[task_name]["normalized_labels"].extend(labels_norm.cpu().numpy().tolist())
        else:
          preds = outputs[task_name][mask].argmax(dim=1)
          labels = raw_labels[mask, task_idx].long()
          val_predictions[task_name]["preds"].extend(preds.cpu().numpy().tolist())
          val_predictions[task_name]["labels"].extend(labels.cpu().numpy().tolist())

  task_reports = {}
  aggregate_score = 0.0
  scored_tasks = 0
  for task_name, values in val_predictions.items():
    if not values["labels"]:
      continue

    metric_name, metric_value, report = _metric_from_preds(values["labels"], values["preds"], task_metas[task_name]["dtype"])
    if task_metas[task_name]["dtype"] == "float":
      normalized_mae = mean_absolute_error(values["normalized_labels"], values["normalized_preds"])
      selection_metric = -normalized_mae
      report["normalized_mae"] = normalized_mae
    else:
      selection_metric = metric_value

    task_reports[task_name] = {
      "metric_name": metric_name,
      "metric_value": metric_value,
      "selection_metric": selection_metric,
      "report": report,
    }
    aggregate_score += selection_metric
    scored_tasks += 1

  aggregate_score /= max(1, scored_tasks)
  summary_parts = []
  for task_name in sorted(task_reports):
    report = task_reports[task_name]["report"]
    if task_metas[task_name]["dtype"] in ("bool", "int"):
      summary_parts.append(
        f"{task_name}:ACC={report['acc']:.4f} F1={report['f1']:.4f}"
      )
    else:
      summary_parts.append(
        f"{task_name}:MAE={report['mae']:.4f} RMSE={report['rmse']:.4f}"
      )
  print(f"Train Loss: {total_loss / len(train_loader):.4f} | Val " + " ".join(summary_parts))

  if aggregate_score > best_metric:
    best_metric = aggregate_score
    stale = 0
    model_ref = unwrap_model(model)
    best_state = {
      "adapter": {k: v.cpu() for k, v in model_ref.adapter.state_dict().items()},
      "pool": {k: v.cpu() for k, v in model_ref.pool.state_dict().items()},
      "heads": {task_name: {k: v.cpu() for k, v in head.state_dict().items()} for task_name, head in model_ref.heads.items()},
      "aggregate_score": aggregate_score,
      "task_reports": task_reports,
    }
  else:
    stale += 1
    if stale >= PATIENCE:
      print("Early stopping.")
      break

if best_state is not None:
  model_ref = unwrap_model(model)
  model_ref.adapter.load_state_dict(best_state["adapter"])
  model_ref.pool.load_state_dict(best_state["pool"])
  for task_name, state_dict in best_state["heads"].items():
    model_ref.heads[task_name].load_state_dict(state_dict)

model_ref = unwrap_model(model)
out_path = "./prostt5_multitask_adapter_best.pt"
torch.save(
  {
    "adapter_state_dict": model_ref.adapter.state_dict(),
    "pool_state_dict": model_ref.pool.state_dict(),
    "head_state_dicts": {task_name: head.state_dict() for task_name, head in model_ref.heads.items()},
    "config": {
      "embed_dim": embed_dim,
      "adapter_dim": ADAPTER_DIM,
      "dropout": DROPOUT,
      "attn_pool_hidden": ATTN_POOL_HIDDEN,
      "model_name": MODEL_NAME,
      "tokenized_data_path": str(TRAIN_CACHE_PATH),
      "task_names": task_order,
      "task_metas": task_metas,
      "task_output_dims": task_output_dims,
      "regression_mean": regression_means,
      "regression_std": regression_stds,
      "best_aggregate_score": best_state["aggregate_score"] if best_state else None,
      "best_task_reports": best_state["task_reports"] if best_state else None,
    },
  },
  out_path,
)
print(f"Saved best shared adapter+heads -> {out_path}")
