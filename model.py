"""
Shared model and data utilities for multitask ProstT5 training and validation.
"""

import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

from config import ADAPTER_DIM, ATTN_POOL_HIDDEN, DROPOUT


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
  def __init__(self, dataset, batch_size, shuffle=False, seed=0, sample_weights=None, max_tokens_per_batch=None):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.sample_weights = sample_weights
    self.max_tokens_per_batch = max_tokens_per_batch
    self.epoch = 0
    self.pool_size = batch_size * 50
    self.num_samples = len(dataset)

  def _pack_batches(self, indices):
    if self.max_tokens_per_batch is None:
      return [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    batches = []
    current_batch = []
    current_max_len = 0

    for idx in indices:
      sample_len = self.dataset.samples[idx]["length"]
      proposed_max_len = max(current_max_len, sample_len)
      proposed_batch_size = len(current_batch) + 1
      would_exceed_tokens = proposed_max_len * proposed_batch_size > self.max_tokens_per_batch

      if current_batch and (would_exceed_tokens or len(current_batch) >= self.batch_size):
        batches.append(current_batch)
        current_batch = []
        current_max_len = 0
        proposed_max_len = sample_len

      current_batch.append(idx)
      current_max_len = proposed_max_len

    if current_batch:
      batches.append(current_batch)

    return batches

  def __iter__(self):
    if not self.shuffle:
      indices = list(range(len(self.dataset)))
      indices.sort(key=lambda idx: self.dataset.samples[idx]["length"])
      batches = self._pack_batches(indices)
      return iter(batches)

    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch)
    self.epoch += 1

    sampled_indices = torch.multinomial(
      self.sample_weights,
      self.num_samples,
      replacement=True,
      generator=generator,
    ).tolist()

    batches = []
    for start in range(0, len(sampled_indices), self.pool_size):
      pool = sampled_indices[start:start + self.pool_size]
      pool.sort(key=lambda idx: self.dataset.samples[idx]["length"])
      batches.extend(self._pack_batches(pool))

    order = torch.randperm(len(batches), generator=generator).tolist()
    return iter([batches[idx] for idx in order])

  def __len__(self):
    return math.ceil(self.num_samples / self.batch_size)


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
    for param in self.base.parameters():
      param.requires_grad = False
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


def unwrap_model(model):
  return model._orig_mod if hasattr(model, "_orig_mod") else model


def output_dim_from_meta(meta, labels, mask):
  if meta["dtype"] == "float":
    return 1

  if meta["num_classes"] is not None:
    return int(meta["num_classes"])

  observed = labels[mask]
  if observed.numel() == 0:
    raise ValueError(f"Task '{meta['task_name']}' has no observed labels in train split.")
  return int(observed.max().item()) + 1


def collate_multitask_batch(batch, pad_token_id):
  input_ids = [sample["input_ids"] for sample in batch]
  padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
  attention_mask = padded_ids.ne(pad_token_id).long()
  raw_labels = torch.stack([sample["raw_labels"] for sample in batch])
  normalized_labels = torch.stack([sample["normalized_labels"] for sample in batch])
  label_mask = torch.stack([sample["label_mask"] for sample in batch])
  return padded_ids, attention_mask, raw_labels, normalized_labels, label_mask
