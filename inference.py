"""
Run multitask ProstT5 adapter inference from a raw amino-acid sequence or FASTA file.
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

import torch
from neurosnap.sequence.align import read_msa
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
VALID_SEQUENCE_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYUZOBX]+$")
DEFAULT_INFERENCE_CHECKPOINT = Path("checkpoints/prostt5_multitask_adapter_best_2026-05-27_seed_1.pt")


def preprocess_sequence(seq: str) -> str:
  seq = re.sub(r"[UZOB]", "X", seq.upper())
  spaced = " ".join(list(seq))
  return "<AA2fold> " + spaced


def normalize_sequence(seq: str) -> str:
  sequence = re.sub(r"\s+", "", seq).upper()
  if not sequence:
    raise ValueError("Sequence is empty.")
  if VALID_SEQUENCE_PATTERN.fullmatch(sequence) is None:
    raise ValueError("Sequence contains invalid amino-acid characters.")
  return sequence


def load_fasta_sequences(fasta_path: Path):
  if not fasta_path.exists():
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")

  records = []
  for name, sequence in read_msa(str(fasta_path), allow_chars="UZOBX"):
    records.append(
      {
        "name": name,
        "sequence": normalize_sequence(sequence),
        "source": str(fasta_path),
      }
    )

  if not records:
    raise ValueError(f"No sequences found in FASTA: {fasta_path}")
  return records


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

  if DEFAULT_INFERENCE_CHECKPOINT.exists():
    return DEFAULT_INFERENCE_CHECKPOINT

  candidates = sorted(Path("checkpoints").glob("prostt5_multitask_adapter_best*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
  if not candidates:
    candidates = sorted(Path(".").glob("prostt5_multitask_adapter_best*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
  if not candidates:
    raise FileNotFoundError(
      f"Default checkpoint not found at '{DEFAULT_INFERENCE_CHECKPOINT}' and no fallback checkpoint matched "
      "'checkpoints/prostt5_multitask_adapter_best*.pt' or './prostt5_multitask_adapter_best*.pt'."
    )
  return candidates[0]


def should_use_local_files_only(args) -> bool:
  """Return whether Hugging Face assets should be loaded strictly from local cache."""
  env_value = os.getenv("PROT2PROP_LOCAL_FILES_ONLY", "")
  env_enabled = env_value.strip().lower() in {"1", "true", "yes", "on"}
  return bool(getattr(args, "local_files_only", False) or env_enabled)


def load_model_and_tokenizer(checkpoint_path: Path, local_files_only: bool = False):
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

  tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, local_files_only=local_files_only)
  base_model = T5EncoderModel.from_pretrained(model_name, local_files_only=local_files_only).to(DEVICE)
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


def iter_sequence_batches(records, batch_size: int, max_tokens_per_batch: int | None):
  sorted_indices = sorted(range(len(records)), key=lambda idx: len(records[idx]["sequence"]))
  current_batch = []
  current_max_len = 0

  for idx in sorted_indices:
    sequence_len = len(records[idx]["sequence"])
    proposed_max_len = max(current_max_len, sequence_len)
    proposed_batch_size = len(current_batch) + 1
    exceeds_token_budget = max_tokens_per_batch is not None and proposed_max_len * proposed_batch_size > max_tokens_per_batch

    if current_batch and (len(current_batch) >= batch_size or exceeds_token_budget):
      yield current_batch
      current_batch = []
      current_max_len = 0
      proposed_max_len = sequence_len

    current_batch.append(idx)
    current_max_len = proposed_max_len

  if current_batch:
    yield current_batch


def _prediction_from_logits(logits, task_idx, task_name, task_metas, regression_means, regression_stds):
  meta = task_metas[task_name]
  logits = logits.detach().float().cpu()
  if meta["dtype"] == "float":
    pred_norm = float(logits.squeeze(-1).item())
    pred = pred_norm
    if regression_means is not None and regression_stds is not None:
      pred = pred_norm * float(regression_stds[task_idx]) + float(regression_means[task_idx])
    return {
      "type": "regression",
      "value": pred,
      "normalized_value": pred_norm,
    }

  probs = torch.softmax(logits, dim=-1)
  task_prediction = {
    "type": "classification",
    "predicted_class": int(torch.argmax(probs).item()),
    "probabilities": probs.tolist(),
  }
  if logits.numel() == 2:
    task_prediction["positive_probability"] = float(probs[1].item())
  return task_prediction


def predict_sequences(
  records, model, tokenizer, task_order, task_metas, regression_means, regression_stds, batch_size: int, max_tokens_per_batch: int | None
):
  predictions = [None] * len(records)

  for batch_indices in iter_sequence_batches(records, batch_size=batch_size, max_tokens_per_batch=max_tokens_per_batch):
    batch_sequences = [preprocess_sequence(records[idx]["sequence"]) for idx in batch_indices]
    encoded = tokenizer(
      batch_sequences,
      add_special_tokens=True,
      padding="longest",
      return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(encoded.input_ids, encoded.attention_mask)

    for row_idx, record_idx in enumerate(batch_indices):
      task_predictions = {}
      for task_idx, task_name in enumerate(task_order):
        task_predictions[task_name] = _prediction_from_logits(
          outputs[task_name][row_idx],
          task_idx,
          task_name,
          task_metas,
          regression_means,
          regression_stds,
        )
      predictions[record_idx] = task_predictions

  return predictions


def resolve_output_csv_path(output_csv: str | None, fasta_path: str | None) -> Path:
  if output_csv:
    return Path(output_csv)
  if fasta_path:
    fasta = Path(fasta_path)
    return fasta.with_suffix(f"{fasta.suffix}.predictions.csv") if fasta.suffix else fasta.with_name(f"{fasta.name}.predictions.csv")
  return Path("inference_predictions.csv")


def _append_prediction_columns(row: dict, task_name: str, prediction: dict):
  if prediction["type"] == "regression":
    row[f"{task_name}_value"] = prediction["value"]
    row[f"{task_name}_normalized_value"] = prediction["normalized_value"]
    return

  row[f"{task_name}_predicted_class"] = prediction["predicted_class"]
  if "positive_probability" in prediction:
    row[f"{task_name}_positive_probability"] = prediction["positive_probability"]
  for class_idx, probability in enumerate(prediction["probabilities"]):
    row[f"{task_name}_probability_class_{class_idx}"] = probability


def write_predictions_csv(output_csv_path: Path, records, predictions, task_order):
  rows = []
  fieldnames = ["name", "sequence"]

  for record, record_predictions in zip(records, predictions):
    row = {
      "name": record["name"],
      "sequence": record["sequence"],
    }
    for task_name in task_order:
      _append_prediction_columns(row, task_name, record_predictions[task_name])
    rows.append(row)

  for row in rows:
    for fieldname in row.keys():
      if fieldname not in fieldnames:
        fieldnames.append(fieldname)

  output_csv_path.parent.mkdir(parents=True, exist_ok=True)
  with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def build_arg_parser():
  parser = argparse.ArgumentParser(description="Run inference with a trained multitask ProstT5 adapter checkpoint.")
  parser.add_argument("--checkpoint", help=f"Path to a saved adapter checkpoint. Defaults to {DEFAULT_INFERENCE_CHECKPOINT}.")
  parser.add_argument(
    "--download-weights",
    action="store_true",
    help="Download the ProstT5 tokenizer and backbone weights into the local Hugging Face cache, then exit.",
  )
  parser.add_argument("--sequence", help="Raw amino-acid sequence to score.")
  parser.add_argument("--fasta", help="Path to a FASTA file containing one or more amino-acid sequences to score.")
  parser.add_argument(
    "--output-csv", help="Path to write a CSV of prediction outputs. Defaults to a path derived from the FASTA input or ./inference_predictions.csv."
  )
  parser.add_argument("--batch-size", type=int, default=16, help="Maximum number of sequences per inference batch when using FASTA input.")
  parser.add_argument(
    "--max-tokens-per-batch",
    type=int,
    default=12000,
    help="Approximate padded token budget per inference batch. Lower this if GPU memory is limited.",
  )
  parser.add_argument(
    "--local-files-only",
    action="store_true",
    help="Load ProstT5 assets only from the local Hugging Face cache. You can also set PROT2PROP_LOCAL_FILES_ONLY=1.",
  )
  return parser


def main():
  parser = build_arg_parser()
  args = parser.parse_args()

  if args.download_weights:
    if args.sequence or args.fasta:
      raise SystemExit("--download-weights cannot be combined with --sequence or --fasta.")
    offload_folder = "/tmp/prostt5_offload"
    print(f"Downloading tokenizer for {MODEL_NAME}...")
    T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    print(f"Downloading backbone weights for {MODEL_NAME}...")
    warm_model = None
    try:
      warm_model = T5EncoderModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        device_map={"": "cpu"},
        max_memory={"cpu": "4GiB"},
        offload_folder=offload_folder,
        offload_state_dict=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
      )
    finally:
      del warm_model
      shutil.rmtree(offload_folder, ignore_errors=True)
    print(f"Downloaded ProstT5 assets for {MODEL_NAME}.")
    return

  provided_inputs = sum(bool(value) for value in (args.sequence, args.fasta))
  if provided_inputs != 1:
    raise SystemExit("Provide exactly one of --sequence or --fasta.")

  if args.batch_size <= 0:
    raise SystemExit("--batch-size must be greater than zero.")
  if args.max_tokens_per_batch is not None and args.max_tokens_per_batch <= 0:
    raise SystemExit("--max-tokens-per-batch must be greater than zero.")

  if args.sequence:
    records = [
      {
        "name": "input_sequence",
        "sequence": normalize_sequence(args.sequence),
        "source": "raw sequence",
      }
    ]
  else:
    records = load_fasta_sequences(Path(args.fasta))

  output_csv_path = resolve_output_csv_path(args.output_csv, args.fasta)
  checkpoint_path = resolve_checkpoint_path(args.checkpoint)
  local_files_only = should_use_local_files_only(args)
  model, tokenizer, task_order, task_metas, regression_means, regression_stds = load_model_and_tokenizer(
    checkpoint_path,
    local_files_only=local_files_only,
  )
  predictions = predict_sequences(
    records,
    model,
    tokenizer,
    task_order,
    task_metas,
    regression_means,
    regression_stds,
    batch_size=args.batch_size,
    max_tokens_per_batch=args.max_tokens_per_batch,
  )
  write_predictions_csv(output_csv_path, records, predictions, task_order)

  print(f"Checkpoint: {checkpoint_path}")
  print(f"Inputs scored: {len(records)}")
  print(f"CSV: {output_csv_path}")
  for record, record_predictions in zip(records, predictions):
    print()
    print(f"Name: {record['name']}")
    print(f"Source: {record['source']}")
    print(f"Sequence length: {len(record['sequence'])}")
    for task_name in task_order:
      prediction = record_predictions[task_name]
      if prediction["type"] == "regression":
        print(f"{task_name}: value={prediction['value']:.4f} (normalized={prediction['normalized_value']:.4f})")
        continue

      if "positive_probability" in prediction:
        print(f"{task_name}: class={prediction['predicted_class']} positive_prob={prediction['positive_probability']:.4f}")
      else:
        probs = ", ".join(f"{prob:.4f}" for prob in prediction["probabilities"])
        print(f"{task_name}: class={prediction['predicted_class']} probs=[{probs}]")


if __name__ == "__main__":
  main()
