"""
Multi-task training script for one shared ProstT5 adapter with task-specific heads,
backed by per-task tokenized train/validation/test caches.
"""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import T5EncoderModel, get_linear_schedule_with_warmup

from config import MODEL_NAME, TOKENIZED_DATA_DIR


### Config & hyperparameters
# Training batch size. Increase until GPU memory or throughput stops improving.
BATCH_SIZE = 32
# Optimizer learning rate for the adapter and task heads.
LR = 1e-4
# Maximum number of training epochs before early stopping cuts the run short.
EPOCHS = 10
# Adapter bottleneck width. Higher values add capacity and compute.
ADAPTER_DIM = 64
# Shared dropout used by the adapter and attention pooling head.
DROPOUT = 0.1
# Hidden width inside the attention pooling projection.
ATTN_POOL_HIDDEN = 256
# AdamW weight decay for trainable parameters.
WEIGHT_DECAY = 1e-2
# Warmup ratio for the linear LR scheduler.
WARMUP_RATIO = 0.05
# Early stopping patience measured in epochs without improvement.
PATIENCE = 3
# Seed used when shuffling bucketed batches each epoch.
BATCH_SAMPLER_SEED = 42


### Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
COMPILE_MODEL = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
USE_FUSED_ADAMW = DEVICE.type == "cuda"


class MultiTaskTokenizedDataset(Dataset):
  def __init__(self, task_payloads):
    self.samples = []
    self.task_meta = {}
    self.pad_token_id = None

    for task_name, payload in task_payloads.items():
      self.task_meta[task_name] = payload["meta"]
      if self.pad_token_id is None:
        self.pad_token_id = payload["config"]["pad_token_id"]

      split = payload["split"]
      for local_idx, length in enumerate(split["lengths"]):
        self.samples.append(
          {
            "task_name": task_name,
            "input_ids": split["input_ids"][local_idx],
            "label": split["labels"][local_idx],
            "length": int(length),
          }
        )

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


class MultiTaskBatchSampler(Sampler):
  def __init__(self, dataset, batch_size, shuffle, seed):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.epoch = 0

  def __iter__(self):
    rng = random.Random(self.seed + self.epoch)
    if self.shuffle:
      self.epoch += 1

    indices_by_task = defaultdict(list)
    for idx, sample in enumerate(self.dataset.samples):
      indices_by_task[sample["task_name"]].append(idx)

    batches_by_task = {}
    max_batches = 0
    for task_name, indices in indices_by_task.items():
      if self.shuffle:
        rng.shuffle(indices)
      # Keep batches homogeneous by task while still minimizing padding.
      indices.sort(key=lambda idx: self.dataset.samples[idx]["length"])
      batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
      if self.shuffle:
        rng.shuffle(batches)
      batches_by_task[task_name] = batches
      max_batches = max(max_batches, len(batches))

    all_batches = []
    task_names = sorted(batches_by_task.keys())
    for task_name in task_names:
      task_batches = batches_by_task[task_name]
      for batch_idx in range(max_batches):
        # Oversample smaller tasks so each task contributes the same number of batches per epoch.
        all_batches.append(task_batches[batch_idx % len(task_batches)])

    if self.shuffle:
      rng.shuffle(all_batches)
    return iter(all_batches)

  def __len__(self):
    indices_by_task = Counter(sample["task_name"] for sample in self.dataset.samples)
    return max(math.ceil(count / self.batch_size) for count in indices_by_task.values()) * len(indices_by_task)


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
  def __init__(self, base_model, task_metas, train_labels_by_task, embed_dim, adapter_dim=ADAPTER_DIM, dropout=DROPOUT):
    super().__init__()
    self.base = base_model
    for p in self.base.parameters():
      p.requires_grad = False
    self.adapter = Adapter(embed_dim, adapter_dim, dropout_prob=dropout)
    self.pool = AttnPool(embed_dim, hidden=ATTN_POOL_HIDDEN, dropout=dropout)
    self.heads = nn.ModuleDict()

    for task_name, meta in task_metas.items():
      output_dim = _output_dim_from_meta(meta, train_labels_by_task[task_name])
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

  def forward(self, input_ids, attention_mask, task_name):
    pooled = self.encode(input_ids, attention_mask)
    return self.heads[task_name](pooled)


def _build_loss(meta: Dict[str, str], train_labels):
  dtype = meta["dtype"]
  loss_name = (meta["loss"] or "").lower()

  if dtype in ("bool", "int"):
    if dtype == "bool" and (meta["num_classes"] in (None, 2)):
      counts = Counter(int(x) for x in train_labels)
      n0, n1 = counts.get(0, 0), counts.get(1, 0)
      total = n0 + n1
      w0 = total / (2.0 * max(1, n0))
      w1 = total / (2.0 * max(1, n1))
      weights = torch.tensor([w0, w1], dtype=torch.float, device=DEVICE)
      return nn.CrossEntropyLoss(weight=weights)
    return nn.CrossEntropyLoss()

  if loss_name in ("mae", "l1"):
    return nn.L1Loss()
  return nn.MSELoss()


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


def _output_dim_from_meta(meta: Dict[str, str], train_labels: torch.Tensor) -> int:
  if meta["dtype"] == "float":
    return 1

  if meta["num_classes"] is not None:
    return int(meta["num_classes"])

  labels = {int(x) for x in train_labels.view(-1).tolist()}
  return max(labels) + 1


def _collate_batch(batch, pad_token_id):
  task_names = [sample["task_name"] for sample in batch]
  task_name = task_names[0]
  if any(name != task_name for name in task_names):
    raise ValueError("Each batch must contain samples from exactly one task")

  input_ids = [sample["input_ids"] for sample in batch]
  labels = torch.stack([sample["label"] for sample in batch])
  # Pad only to the longest sequence in this batch.
  padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
  attention_mask = padded_ids.ne(pad_token_id).long()
  return task_name, padded_ids, attention_mask, labels


print("Loading tokenized multi-task splits")
tokenized_paths = sorted(Path(TOKENIZED_DATA_DIR).glob("*_prostt5_tokens.pt"))
if not tokenized_paths:
  raise FileNotFoundError(f"No tokenized task files found under {TOKENIZED_DATA_DIR}. Run tokenize_data.py first.")

train_payloads = {}
val_payloads = {}
for path in tokenized_paths:
  payload = torch.load(path, map_location="cpu")
  task_name = payload["meta"]["task_name"]
  train_payloads[task_name] = {
    "meta": payload["meta"],
    "config": payload["config"],
    "split": payload["splits"]["train"],
  }
  val_payloads[task_name] = {
    "meta": payload["meta"],
    "config": payload["config"],
    "split": payload["splits"]["validation"],
  }

train_ds = MultiTaskTokenizedDataset(train_payloads)
val_ds = MultiTaskTokenizedDataset(val_payloads)
pad_token_id = train_ds.pad_token_id

print(f"Loaded {len(train_payloads)} tasks from {TOKENIZED_DATA_DIR}")
for task_name in sorted(train_payloads):
  meta = train_payloads[task_name]["meta"]
  train_rows = train_payloads[task_name]["split"]["labels"].shape[0]
  val_rows = val_payloads[task_name]["split"]["labels"].shape[0]
  print(f"Task={task_name} dtype={meta['dtype']} head={meta['head_type']} loss={meta['loss']} train={train_rows} val={val_rows}")

base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == "cuda":
  base_model.bfloat16()

train_loader = DataLoader(
  train_ds,
  batch_sampler=MultiTaskBatchSampler(train_ds, BATCH_SIZE, shuffle=True, seed=BATCH_SAMPLER_SEED),
  collate_fn=lambda batch: _collate_batch(batch, pad_token_id),
  pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
  val_ds,
  batch_sampler=MultiTaskBatchSampler(val_ds, BATCH_SIZE, shuffle=False, seed=BATCH_SAMPLER_SEED),
  collate_fn=lambda batch: _collate_batch(batch, pad_token_id),
  pin_memory=PIN_MEMORY,
)

print("Initializing model")
task_metas = {task_name: payload["meta"] for task_name, payload in train_payloads.items()}
train_labels_by_task = {task_name: payload["split"]["labels"] for task_name, payload in train_payloads.items()}
embed_dim = base_model.config.d_model
model = MultiTaskAdapterModel(base_model, task_metas, train_labels_by_task, embed_dim=embed_dim, adapter_dim=ADAPTER_DIM, dropout=DROPOUT).to(DEVICE)

if COMPILE_MODEL and hasattr(torch, "compile"):
  print("Compiling model")
  try:
    model = torch.compile(model)
  except Exception as exc:
    print(f"torch.compile unavailable, continuing without compile: {exc}")

criteria = {
  task_name: _build_loss(meta, train_labels_by_task[task_name].view(-1).tolist())
  for task_name, meta in task_metas.items()
}
optimizer = torch.optim.AdamW(
  [{"params": model.adapter.parameters()}, {"params": model.pool.parameters()}, {"params": model.heads.parameters()}],
  lr=LR,
  weight_decay=WEIGHT_DECAY,
  fused=USE_FUSED_ADAMW,
)
trainable_params = list(model.adapter.parameters()) + list(model.pool.parameters()) + list(model.heads.parameters())

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_metric = -float("inf")
stale = 0
best_state = None

for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0

  for task_name, input_ids, attn_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
    input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
    attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
    labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
      preds = model(input_ids, attn_mask, task_name)
      loss = criteria[task_name](preds, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  model.eval()
  val_predictions = {task_name: {"preds": [], "labels": []} for task_name in task_metas}
  with torch.no_grad():
    for task_name, input_ids, attn_mask, labels in val_loader:
      input_ids = input_ids.to(DEVICE, non_blocking=PIN_MEMORY)
      attn_mask = attn_mask.to(DEVICE, non_blocking=PIN_MEMORY)
      labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(input_ids, attn_mask, task_name)

      if task_metas[task_name]["dtype"] == "float":
        val_predictions[task_name]["preds"].extend(outputs.squeeze(-1).float().cpu().numpy().tolist())
        val_predictions[task_name]["labels"].extend(labels.squeeze(-1).float().cpu().numpy().tolist())
      else:
        val_predictions[task_name]["preds"].extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        val_predictions[task_name]["labels"].extend(labels.cpu().numpy().tolist())

  task_reports = {}
  aggregate_score = 0.0
  for task_name, values in val_predictions.items():
    metric_name, metric_value, report = _metric_from_preds(values["labels"], values["preds"], task_metas[task_name]["dtype"])
    task_reports[task_name] = {
      "metric_name": metric_name,
      "metric_value": metric_value,
      "report": report,
    }
    aggregate_score += -metric_value if task_metas[task_name]["dtype"] == "float" else metric_value

  aggregate_score /= len(task_reports)
  summary_parts = []
  for task_name in sorted(task_reports):
    metric_name = task_reports[task_name]["metric_name"].upper()
    metric_value = task_reports[task_name]["metric_value"]
    summary_parts.append(f"{task_name}:{metric_name}={metric_value:.4f}")
  print(f"Train Loss: {total_loss / len(train_loader):.4f} | Val " + " ".join(summary_parts))

  if aggregate_score > best_metric:
    best_metric = aggregate_score
    stale = 0
    best_state = {
      "adapter": {k: v.cpu() for k, v in model.adapter.state_dict().items()},
      "pool": {k: v.cpu() for k, v in model.pool.state_dict().items()},
      "heads": {task_name: {k: v.cpu() for k, v in head.state_dict().items()} for task_name, head in model.heads.items()},
      "aggregate_score": aggregate_score,
      "task_reports": task_reports,
    }
  else:
    stale += 1
    if stale >= PATIENCE:
      print("Early stopping.")
      break

if best_state is not None:
  model.adapter.load_state_dict(best_state["adapter"])
  model.pool.load_state_dict(best_state["pool"])
  for task_name, state_dict in best_state["heads"].items():
    model.heads[task_name].load_state_dict(state_dict)

out_path = "./prostt5_multitask_adapter_best.pt"
torch.save(
  {
    "adapter_state_dict": model.adapter.state_dict(),
    "pool_state_dict": model.pool.state_dict(),
    "head_state_dicts": {task_name: head.state_dict() for task_name, head in model.heads.items()},
    "config": {
      "embed_dim": embed_dim,
      "adapter_dim": ADAPTER_DIM,
      "dropout": DROPOUT,
      "attn_pool_hidden": ATTN_POOL_HIDDEN,
      "model_name": MODEL_NAME,
      "tokenized_data_dir": str(TOKENIZED_DATA_DIR),
      "task_names": sorted(task_metas.keys()),
      "task_metas": task_metas,
      "best_aggregate_score": best_state["aggregate_score"] if best_state else None,
      "best_task_reports": best_state["task_reports"] if best_state else None,
    },
  },
  out_path,
)
print(f"Saved best shared adapter+heads -> {out_path}")
