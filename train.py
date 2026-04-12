"""
Single-task training script for a frozen ProstT5 encoder with a lightweight adapter
and task head, backed by pre-tokenized train/validation/test tensors.
"""

import math
from collections import Counter
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import T5EncoderModel, get_linear_schedule_with_warmup

from config import MODEL_NAME, TASK_NAME, TOKENIZED_DATA_PATH

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
COMPILE_MODEL = DEVICE.type == "cuda"


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
    h = self.proj(x)
    scores = self.context(h).squeeze(-1)
    scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=1)
    return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


class TaskAdapterModel(nn.Module):
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

    out = self.base(input_ids=input_ids, attention_mask=attention_mask)
    token_repr = out.last_hidden_state
    adapted = token_repr + self.adapter(token_repr)
    pooled = self.pool(adapted, attention_mask)
    return self.head(pooled)


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


print("Loading tokenized splits")
if not TOKENIZED_DATA_PATH.exists():
  raise FileNotFoundError(f"Tokenized data not found at {TOKENIZED_DATA_PATH}. Run tokenize_data.py first.")

payload = torch.load(TOKENIZED_DATA_PATH, map_location="cpu")
meta = payload["meta"]
splits = payload["splits"]

print(f"Task={meta['task_name']} dtype={meta['dtype']} head={meta['head_type']} loss={meta['loss']}")
print(
  "Rows: "
  f"train={splits['train']['labels'].shape[0]} "
  f"val={splits['validation']['labels'].shape[0]} "
  f"test={splits['test']['labels'].shape[0]}"
)

base_model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
if DEVICE.type == "cuda":
  base_model.bfloat16()

train_ds = TensorDataset(
  splits["train"]["input_ids"],
  splits["train"]["attention_mask"],
  splits["train"]["labels"],
)
val_ds = TensorDataset(
  splits["validation"]["input_ids"],
  splits["validation"]["attention_mask"],
  splits["validation"]["labels"],
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

print("Initializing model")
embed_dim = base_model.config.d_model
output_dim = _output_dim_from_meta(meta, splits["train"]["labels"])
model = TaskAdapterModel(base_model, embed_dim, output_dim=output_dim, adapter_dim=64).to(DEVICE)

if COMPILE_MODEL and hasattr(torch, "compile"):
  print("Compiling model")
  try:
    model = torch.compile(model)
  except Exception as exc:
    print(f"torch.compile unavailable, continuing without compile: {exc}")

criterion = _build_loss(meta, splits["train"]["labels"].view(-1).tolist())
optimizer = torch.optim.AdamW([{"params": model.adapter.parameters()}, {"params": model.head.parameters()}], lr=LR, weight_decay=1e-2, fused=True)
trainable_params = list(model.adapter.parameters()) + list(model.head.parameters())

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
    input_ids = input_ids.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
      preds = model(input_ids, attn_mask)
      loss = criterion(preds, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()
    scheduler.step()
    total_loss += loss.item()

  model.eval()
  all_preds, all_labels = [], []
  with torch.no_grad():
    for input_ids, attn_mask, labels in val_loader:
      input_ids = input_ids.to(DEVICE)
      attn_mask = attn_mask.to(DEVICE)
      labels = labels.to(DEVICE)
      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(input_ids, attn_mask)

      if is_regression:
        all_preds.extend(outputs.squeeze(-1).float().cpu().numpy().tolist())
        all_labels.extend(labels.squeeze(-1).float().cpu().numpy().tolist())
      else:
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

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
      "tokenized_data_path": str(TOKENIZED_DATA_PATH),
      "task_name": TASK_NAME,
      "task_dtype": meta["dtype"],
      "head_type": meta["head_type"],
      "loss": meta["loss"],
      "best_metric_name": best_state["metric_name"] if best_state else None,
      "best_metric_value": best_state["metric_value"] if best_state else None,
    },
  },
  out_path,
)
print(f"Saved best adapter+head -> {out_path}")
