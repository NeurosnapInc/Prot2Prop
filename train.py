"""
Single-task training script for a frozen ProstT5 encoder with a lightweight adapter
and task head, backed by the DuckDB aggregation format.
"""

import math
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

import duckdb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer, get_linear_schedule_with_warmup

# =====================
# Config
# =====================
MODEL_NAME = "Rostlab/ProstT5"  # ProstT5 encoder
AGGREGATED_DB_PATH = "data/aggregated/aggregated.duckdb"
TASK_NAME = "solubility"

SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
MAX_LENGTH = 1024 * 2
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # TODO TMP


def _validate_split_fractions():
  # Require fractions to sum to 1.0 for deterministic split slicing.
  # Keeping this strict avoids accidental silent data loss from bad split math.
  total = TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION
  if abs(total - 1.0) > 1e-8:
    raise ValueError(f"Split fractions must sum to 1.0, got {total}")


class ProteinTaskDataset(Dataset):
  # Thin dataset wrapper around normalized {sequence, label} rows.
  def __init__(self, rows: List[Dict[str, float]]):
    self.rows = rows

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, idx):
    return self.rows[idx]


# =====================
# Adapter module
# =====================
class Adapter(nn.Module):
  def __init__(self, input_dim, adapter_dim=64, dropout_prob=0.1):
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
  def __init__(self, d_model, hidden=256, dropout=0.1):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(d_model, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.context = nn.Linear(hidden, 1, bias=False)

  def forward(self, x, mask):
    # x: [B,L,d], mask: [B,L] (1 for tokens to keep)
    h = self.proj(x)
    scores = self.context(h).squeeze(-1)
    scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=1)
    pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
    return pooled


class TaskAdapterModel(nn.Module):
  # Shared frozen encoder + adapter, with a configurable task head.
  def __init__(self, base_model, embed_dim, output_dim, adapter_dim=64, dropout=0.1):
    super().__init__()
    self.base = base_model
    for p in self.base.parameters():
      p.requires_grad = False
    self.adapter = Adapter(embed_dim, adapter_dim, dropout_prob=dropout)
    self.pool = AttnPool(embed_dim, hidden=256, dropout=dropout)
    self.head = nn.Sequential(
      nn.LayerNorm(embed_dim),
      nn.Linear(embed_dim, output_dim),
    )

  def forward(self, input_ids, attention_mask):
    if input_ids.dtype != torch.long:
      input_ids = input_ids.long()
    if attention_mask.dtype not in (torch.long, torch.int64, torch.bool):
      attention_mask = attention_mask.long()

    autocast_enabled = input_ids.is_cuda
    with torch.amp.autocast("cuda", enabled=autocast_enabled):
      out = self.base(input_ids=input_ids, attention_mask=attention_mask)
      token_repr = out.last_hidden_state

    token_repr = token_repr.float()
    adapted = token_repr + self.adapter(token_repr)
    pooled = self.pool(adapted, attention_mask)
    return self.head(pooled)


# =====================
# Data helpers
# =====================
def preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  spaced = " ".join(list(seq))
  return "<AA2fold> " + spaced


def _label_from_dtype(label: float, dtype: str):
  # DB labels are float; cast to task dtype as directed by tasks.dtype.
  if dtype == "bool":
    return 1 if float(label) > 0.5 else 0
  if dtype == "int":
    return int(round(float(label)))
  return float(label)


def _load_task_rows_from_db(db_path: str, task_name: str):
  # Pull task metadata and samples from DuckDB.
  # `tasks` config defines how labels are interpreted during training.
  con = duckdb.connect(db_path)
  try:
    task_row = con.execute(
      """
      SELECT task_name, dtype, head_type, num_classes, loss
      FROM tasks
      WHERE task_name = ?
      """,
      [task_name],
    ).fetchone()
    if task_row is None:
      raise ValueError(f"Task '{task_name}' not found in tasks table at {db_path}")

    rows = con.execute(
      """
      SELECT sequence, label
      FROM samples
      WHERE task_name = ?
      """,
      [task_name],
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
    # Keep rows minimal and clean before split/materialization in torch Dataset.
    sequence = (sequence or "").strip()
    if not sequence:
      continue
    normalized.append({"sequence": sequence, "label": _label_from_dtype(label, meta["dtype"])})

  if not normalized:
    raise ValueError(f"No usable samples found for task '{task_name}' in {db_path}")

  return meta, normalized


def _split_rows(rows: List[Dict[str, float]], seed: int) -> Dict[str, List[Dict[str, float]]]:
  # Deterministic random split driven by SPLIT_SEED.
  # We intentionally do this at train time because split is not stored in DB.
  indices = list(range(len(rows)))
  rng = random.Random(seed)
  rng.shuffle(indices)

  n_total = len(indices)
  n_train = int(TRAIN_FRACTION * n_total)
  n_val = int(VAL_FRACTION * n_total)
  n_test = n_total - n_train - n_val

  train_idx = indices[:n_train]
  val_idx = indices[n_train:n_train + n_val]
  test_idx = indices[n_train + n_val:n_train + n_val + n_test]

  return {
    "train": [rows[i] for i in train_idx],
    "validation": [rows[i] for i in val_idx],
    "test": [rows[i] for i in test_idx],
  }


def collate_fn(batch, tokenizer, dtype):
  sequences = [preprocess_sequence(item["sequence"]) for item in batch]
  enc = tokenizer(sequences, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

  # Classification heads consume integer class ids; regression consumes float targets.
  if dtype in ("bool", "int"):
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
  else:
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float).unsqueeze(-1)

  return enc["input_ids"], enc["attention_mask"], labels


def _build_loss(meta: Dict[str, str], train_labels: List[float]):
  # Choose criterion from task metadata; default to sane behavior by dtype.
  # Note: `loss` column only changes regression between MSE and L1 for now.
  dtype = meta["dtype"]
  loss_name = (meta["loss"] or "").lower()

  if dtype in ("bool", "int"):
    if dtype == "bool" and (meta["num_classes"] in (None, 2)):
      # Keep inverse-frequency weighting for binary imbalance.
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
  # Return an optimization metric plus reportable metrics.
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


def _output_dim_from_meta(meta: Dict[str, str], rows: List[Dict[str, float]]) -> int:
  dtype = meta["dtype"]
  if dtype == "float":
    # Single scalar regression output.
    return 1

  if meta["num_classes"] is not None:
    return int(meta["num_classes"])

  # Fallback if num_classes missing for classification tasks.
  labels = {int(r["label"]) for r in rows}
  return max(labels) + 1


# =====================
# Train
# =====================
print("Loading task data from DuckDB")
_validate_split_fractions()
meta, rows = _load_task_rows_from_db(AGGREGATED_DB_PATH, TASK_NAME)
splits = _split_rows(rows, SPLIT_SEED)

print(f"Task={meta['task_name']} dtype={meta['dtype']} head={meta['head_type']} loss={meta['loss']}")
print(f"Rows: train={len(splits['train'])} val={len(splits['validation'])} test={len(splits['test'])}")

if len(splits["train"]) == 0 or len(splits["validation"]) == 0:
  raise ValueError("Train/validation split is empty; adjust dataset size or split fractions.")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == "cuda":
  base_model.half()

train_ds = ProteinTaskDataset(splits["train"])
val_ds = ProteinTaskDataset(splits["validation"])

train_loader = DataLoader(
  train_ds,
  batch_size=BATCH_SIZE,
  shuffle=True,
  collate_fn=lambda x: collate_fn(x, tokenizer, meta["dtype"]),
)
val_loader = DataLoader(
  val_ds,
  batch_size=BATCH_SIZE,
  shuffle=False,
  collate_fn=lambda x: collate_fn(x, tokenizer, meta["dtype"]),
)

print("Initializing model")
embed_dim = base_model.config.d_model
output_dim = _output_dim_from_meta(meta, splits["train"])
# Adapter/head remain trainable; encoder stays frozen.
model = TaskAdapterModel(base_model, embed_dim, output_dim=output_dim, adapter_dim=64).to(DEVICE)
model.float()

criterion = _build_loss(meta, [x["label"] for x in splits["train"]])
optimizer = torch.optim.AdamW([{"params": model.adapter.parameters()}, {"params": model.head.parameters()}], lr=LR, weight_decay=1e-2)

num_training_steps = len(train_loader) * EPOCHS
num_warmup_steps = int(0.05 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

is_regression = meta["dtype"] == "float"
best_metric = float("inf") if is_regression else -1.0
patience = 3
stale = 0
best_state = None

for epoch in range(EPOCHS):
  model.train()
  total_loss = 0.0

  for input_ids, attn_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
    input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)

    preds = model(input_ids, attn_mask)
    loss = criterion(preds, labels)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  model.eval()
  all_preds, all_labels = [], []
  with torch.no_grad():
    for input_ids, attn_mask, labels in val_loader:
      input_ids, attn_mask, labels = input_ids.to(DEVICE), attn_mask.to(DEVICE), labels.to(DEVICE)
      outputs = model(input_ids, attn_mask)

      if is_regression:
        # Regression heads emit [B,1], so squeeze for metric functions.
        pred_values = outputs.squeeze(-1).cpu().numpy().tolist()
        label_values = labels.squeeze(-1).cpu().numpy().tolist()
        all_preds.extend(pred_values)
        all_labels.extend(label_values)
      else:
        # Classification heads emit logits over classes.
        pred_values = outputs.argmax(dim=1).cpu().numpy().tolist()
        label_values = labels.cpu().numpy().tolist()
        all_preds.extend(pred_values)
        all_labels.extend(label_values)

  metric_name, metric_value, report = _metric_from_preds(all_labels, all_preds, meta["dtype"])
  report_parts = [f"{k.upper()}: {v:.4f}" for k, v in report.items()]
  print(f"Train Loss: {total_loss / len(train_loader):.4f} | " + " ".join(report_parts))

  improved = metric_value < best_metric if is_regression else metric_value > best_metric
  if improved:
    best_metric = metric_value
    stale = 0
    best_state = {
      "adapter": {k: v.cpu() for k, v in model.adapter.state_dict().items()},
      "head": {k: v.cpu() for k, v in model.head.state_dict().items()},
      "metric_name": metric_name,
      "metric_value": metric_value,
    }
  else:
    stale += 1
    if stale >= patience:
      print("Early stopping.")
      break

if best_state is not None:
  model.adapter.load_state_dict(best_state["adapter"])
  model.head.load_state_dict(best_state["head"])

out_path = f"./{TASK_NAME}_prostt5_adapter_best.pt"
torch.save(
  {
    "adapter_state_dict": model.adapter.state_dict(),
    "head_state_dict": model.head.state_dict(),
    "config": {
      "embed_dim": embed_dim,
      "output_dim": output_dim,
      "adapter_dim": 64,
      "model_name": MODEL_NAME,
      "db_path": AGGREGATED_DB_PATH,
      "task_name": TASK_NAME,
      "task_dtype": meta["dtype"],
      "head_type": meta["head_type"],
      "loss": meta["loss"],
      "split_seed": SPLIT_SEED,
      "train_fraction": TRAIN_FRACTION,
      "val_fraction": VAL_FRACTION,
      "test_fraction": TEST_FRACTION,
      "best_metric_name": best_state["metric_name"] if best_state else None,
      "best_metric_value": best_state["metric_value"] if best_state else None,
    },
  },
  out_path,
)
print(f"Saved best adapter+head -> {out_path}")
