import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolutionaryAlignmentLoss(nn.Module):
  def __init__(
    self,
    amino_acid_token_ids=None,
    max_positions_per_sequence: int | None = 64,
    min_positions_per_sequence: int = 3,
    eps: float = 1e-8,
  ):
    super().__init__()
    self.max_positions_per_sequence = max_positions_per_sequence
    self.min_positions_per_sequence = min_positions_per_sequence
    self.eps = eps

    if amino_acid_token_ids is None:
      self.register_buffer("aa_token_ids", torch.empty(0, dtype=torch.long), persistent=False)
    else:
      self.register_buffer("aa_token_ids", torch.tensor(amino_acid_token_ids, dtype=torch.long), persistent=False)

  def sample_positions(self, alignment_mask):
    if self.max_positions_per_sequence is None:
      return alignment_mask

    sampled = torch.zeros_like(alignment_mask, dtype=torch.bool)
    for b in range(alignment_mask.size(0)):
      pos = alignment_mask[b].nonzero(as_tuple=False).flatten()
      if pos.numel() == 0:
        continue
      if pos.numel() > self.max_positions_per_sequence:
        pos = pos[torch.randperm(pos.numel(), device=pos.device)[:self.max_positions_per_sequence]]
      sampled[b, pos] = True
    return sampled

  def wt_nll_from_logits(self, logits, residue_token_ids, valid_mask):
    valid_mask = valid_mask & residue_token_ids.ge(0)
    nll = logits.new_zeros(residue_token_ids.shape, dtype=torch.float)

    if self.aa_token_ids.numel() == 0:
      log_probs = F.log_softmax(logits.float(), dim=-1)
      gathered = log_probs.gather(-1, residue_token_ids.clamp_min(0).unsqueeze(-1)).squeeze(-1)
      nll[valid_mask] = -gathered[valid_mask]
      return nll, valid_mask

    aa_ids = self.aa_token_ids.to(logits.device)
    aa_logits = logits.float().index_select(dim=-1, index=aa_ids)
    aa_log_probs = F.log_softmax(aa_logits, dim=-1)

    full_to_aa = torch.full((logits.size(-1),), -1, dtype=torch.long, device=logits.device)
    full_to_aa[aa_ids] = torch.arange(aa_ids.numel(), device=logits.device)
    aa_targets = full_to_aa[residue_token_ids.clamp_min(0)]

    valid_mask = valid_mask & aa_targets.ge(0)
    gathered = aa_log_probs.gather(-1, aa_targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll[valid_mask] = -gathered[valid_mask]
    return nll, valid_mask

  def forward(self, logits, residue_token_ids, target_alignment_nll, alignment_mask):
    pred_nll, valid_mask = self.wt_nll_from_logits(logits, residue_token_ids, alignment_mask)

    losses = []
    corrs = []

    for b in range(pred_nll.size(0)):
      mask = valid_mask[b]
      if int(mask.sum()) < self.min_positions_per_sequence:
        continue

      pred = pred_nll[b, mask]
      target = target_alignment_nll[b, mask].float()

      pred = pred - pred.mean()
      target = target - target.mean()

      pred_std = pred.pow(2).mean().sqrt()
      target_std = target.pow(2).mean().sqrt()

      if pred_std.item() <= self.eps or target_std.item() <= self.eps:
        continue

      corr = (pred * target).mean() / (pred_std * target_std).clamp_min(self.eps)
      losses.append(1.0 - corr)
      corrs.append(corr.detach())

    if not losses:
      return pred_nll.sum() * 0.0, {"mean_corr": float("nan"), "usable_sequences": 0}

    loss = torch.stack(losses).mean()
    return loss, {
      "mean_corr": float(torch.stack(corrs).mean().cpu()),
      "usable_sequences": len(losses),
    }