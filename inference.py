"""
Run single-sequence inference with a trained multitask ProstT5 adapter checkpoint.
Accepts either a raw amino-acid sequence or a PDB file and prints task predictions.
"""

import argparse
import re
from pathlib import Path

import torch
from transformers import T5EncoderModel, T5Tokenizer

from config import (
  ADAPTER_DIM,
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  MODEL_NAME,
  REGRESSION_HEAD_HIDDEN,
  TASK_ADAPTER_DIM,
)
from model import MultiTaskAdapterModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"

# Standard three-letter residue codes to one-letter sequence.
THREE_TO_ONE = {
  "ALA": "A",
  "ARG": "R",
  "ASN": "N",
  "ASP": "D",
  "CYS": "C",
  "GLN": "Q",
  "GLU": "E",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LEU": "L",
  "LYS": "K",
  "MET": "M",
  "PHE": "F",
  "PRO": "P",
  "SER": "S",
  "THR": "T",
  "TRP": "W",
  "TYR": "Y",
  "VAL": "V",
  "SEC": "U",
  "PYL": "O",
}


def _seq_from_seqres(lines):
  residues = []
  for line in lines:
    if not line.startswith("SEQRES"):
      continue
    parts = line.split()
    for res in parts[4:]:
      residues.append(THREE_TO_ONE.get(res.upper(), "X"))
  return "".join(residues)


def _seq_from_atom(lines):
  residues = []
  seen = set()
  for line in lines:
    if not line.startswith("ATOM"):
      continue
    atom_name = line[12:16].strip()
    if atom_name != "CA":
      continue
    resname = line[17:20].strip().upper()
    chain_id = line[21].strip()
    resseq = line[22:26].strip()
    icode = line[26].strip()
    key = (chain_id, resseq, icode)
    if key in seen:
      continue
    seen.add(key)
    residues.append(THREE_TO_ONE.get(resname, "X"))
  return "".join(residues)


def load_sequence_from_pdb(pdb_path: Path) -> str:
  lines = pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines()
  seq = _seq_from_seqres(lines)
  if not seq:
    seq = _seq_from_atom(lines)
  if not seq:
    raise ValueError(f"Unable to derive sequence from PDB: {pdb_path}")
  return seq


def preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  spaced = " ".join(list(seq))
  return "<AA2fold> " + spaced


def _infer_task_output_dims(checkpoint, task_order):
  task_output_dims = {}
  for task_name in task_order:
    head_state = checkpoint["head_state_dicts"][task_name]
    output_weight = None
    for key, value in head_state.items():
      if key.endswith("weight"):
        output_weight = value
    if output_weight is None:
      raise ValueError(f"Could not infer output dimension for task '{task_name}'.")
    task_output_dims[task_name] = int(output_weight.shape[0])
  return task_output_dims


def resolve_checkpoint_path(checkpoint_arg: str | None) -> Path:
  if checkpoint_arg:
    checkpoint_path = Path(checkpoint_arg)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path

  candidates = sorted(Path(".").glob("prostt5_multitask_adapter_best*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
  if not candidates:
    raise FileNotFoundError("No checkpoint found matching 'prostt5_multitask_adapter_best*.pt'.")
  return candidates[0]


def load_model_and_tokenizer(checkpoint_path: Path):
  checkpoint = torch.load(checkpoint_path, map_location="cpu")
  model_config = checkpoint["config"]
  task_order = model_config["task_names"]
  task_metas = model_config["task_metas"]
  task_output_dims = model_config.get("task_output_dims") or _infer_task_output_dims(checkpoint, task_order)

  model_name = model_config.get("model_name", MODEL_NAME)
  classification_head_hidden = model_config.get("classification_head_hidden", 0)
  regression_head_hidden = model_config.get("regression_head_hidden", 0)
  task_adapter_dim = model_config.get("task_adapter_dim", 0)
  if classification_head_hidden == 0 and regression_head_hidden == 0:
    classification_head_hidden = 0
    regression_head_hidden = 0
  else:
    classification_head_hidden = model_config.get("classification_head_hidden", CLASSIFICATION_HEAD_HIDDEN)
    regression_head_hidden = model_config.get("regression_head_hidden", REGRESSION_HEAD_HIDDEN)
  if task_adapter_dim == 0:
    task_adapter_dim = 0
  else:
    task_adapter_dim = model_config.get("task_adapter_dim", TASK_ADAPTER_DIM)

  tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
  base_model = T5EncoderModel.from_pretrained(model_name).to(DEVICE)
  if DEVICE.type == "cuda":
    base_model.bfloat16()

  model = MultiTaskAdapterModel(
    base_model,
    task_order,
    task_output_dims,
    embed_dim=model_config["embed_dim"],
    task_metas=task_metas,
    adapter_dim=model_config.get("adapter_dim", ADAPTER_DIM),
    task_adapter_dim=task_adapter_dim,
    dropout=model_config.get("dropout", DROPOUT),
    classification_head_hidden=classification_head_hidden,
    regression_head_hidden=regression_head_hidden,
  ).to(DEVICE)

  model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
  for task_name, state_dict in checkpoint.get("task_adapter_state_dicts", {}).items():
    model.task_adapters[task_name].load_state_dict(state_dict)
  model.pool.load_state_dict(checkpoint["pool_state_dict"])
  for task_name, state_dict in checkpoint["head_state_dicts"].items():
    model.heads[task_name].load_state_dict(state_dict)
  model.eval()

  regression_means = model_config.get("regression_mean")
  regression_stds = model_config.get("regression_std")
  return model, tokenizer, task_order, task_metas, regression_means, regression_stds


def predict_sequence(sequence: str, model, tokenizer, task_order, task_metas, regression_means, regression_stds):
  encoded = tokenizer.batch_encode_plus(
    [preprocess_sequence(sequence)],
    add_special_tokens=True,
    padding="longest",
    return_tensors="pt",
  ).to(DEVICE)

  with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
      outputs = model(encoded.input_ids, encoded.attention_mask)

  predictions = {}
  for task_idx, task_name in enumerate(task_order):
    meta = task_metas[task_name]
    logits = outputs[task_name][0].detach().float().cpu()
    if meta["dtype"] == "float":
      pred_norm = float(logits.squeeze(-1).item())
      pred = pred_norm
      if regression_means is not None and regression_stds is not None:
        pred = pred_norm * float(regression_stds[task_idx]) + float(regression_means[task_idx])
      predictions[task_name] = {
        "type": "regression",
        "value": pred,
        "normalized_value": pred_norm,
      }
      continue

    probs = torch.softmax(logits, dim=-1)
    pred_class = int(torch.argmax(probs).item())
    task_prediction = {
      "type": "classification",
      "predicted_class": pred_class,
      "probabilities": probs.tolist(),
    }
    if logits.numel() == 2:
      task_prediction["positive_probability"] = float(probs[1].item())
    predictions[task_name] = task_prediction

  return predictions


def build_arg_parser():
  parser = argparse.ArgumentParser(description="Run inference with a trained multitask ProstT5 adapter checkpoint.")
  parser.add_argument("--checkpoint", help="Path to a saved adapter checkpoint. Defaults to the newest local checkpoint.")
  parser.add_argument("--sequence", help="Raw amino-acid sequence to score.")
  parser.add_argument("--pdb", help="Path to a PDB file to convert into a sequence before scoring.")
  return parser


def main():
  parser = build_arg_parser()
  args = parser.parse_args()

  provided_inputs = sum(bool(value) for value in (args.sequence, args.pdb))
  if provided_inputs != 1:
    raise SystemExit("Provide exactly one of --sequence or --pdb.")

  if args.sequence:
    sequence = args.sequence.strip()
    source = "raw sequence"
  else:
    pdb_path = Path(args.pdb)
    sequence = load_sequence_from_pdb(pdb_path)
    source = str(pdb_path)

  checkpoint_path = resolve_checkpoint_path(args.checkpoint)
  model, tokenizer, task_order, task_metas, regression_means, regression_stds = load_model_and_tokenizer(checkpoint_path)
  predictions = predict_sequence(sequence, model, tokenizer, task_order, task_metas, regression_means, regression_stds)

  print(f"Checkpoint: {checkpoint_path}")
  print(f"Source: {source}")
  print(f"Sequence length: {len(sequence)}")
  for task_name in task_order:
    prediction = predictions[task_name]
    if prediction["type"] == "regression":
      print(
        f"{task_name}: value={prediction['value']:.4f} "
        f"(normalized={prediction['normalized_value']:.4f})"
      )
      continue

    if "positive_probability" in prediction:
      print(
        f"{task_name}: class={prediction['predicted_class']} "
        f"positive_prob={prediction['positive_probability']:.4f}"
      )
    else:
      probs = ", ".join(f"{prob:.4f}" for prob in prediction["probabilities"])
      print(f"{task_name}: class={prediction['predicted_class']} probs=[{probs}]")


if __name__ == "__main__":
  main()
