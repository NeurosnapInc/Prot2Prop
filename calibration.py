"""
Post-hoc calibration utilities for multitask validation outputs.

This module keeps the lightweight calibration logic separate from model training and
raw evaluation so callers can:
- fit an affine regression calibrator `y = a * pred + b`
- tune binary classification thresholds on held-out scores
- build the post-hoc reporting rows used by validation

The helpers operate on plain Python lists and task-metadata dictionaries so they can
be reused from scripts such as `validate.py` without depending on training internals.
"""

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


def fit_linear_regression_calibrator(preds, labels):
  """Fit a 1D affine calibrator `y = slope * pred + intercept`.

  The closed-form solution is used directly:
  `slope = cov(pred, label) / var(pred)` and
  `intercept = mean(label) - slope * mean(pred)`.

  If the calibration predictions are constant, the slope is undefined. In that case
  the calibrator falls back to a constant predictor equal to the mean label value.
  """
  preds_tensor = torch.tensor(preds, dtype=torch.float)
  labels_tensor = torch.tensor(labels, dtype=torch.float)

  pred_mean = preds_tensor.mean()
  label_mean = labels_tensor.mean()
  centered_preds = preds_tensor - pred_mean
  centered_labels = labels_tensor - label_mean
  variance = centered_preds.pow(2).mean()

  if variance.item() == 0.0:
    return 0.0, label_mean.item()

  covariance = (centered_preds * centered_labels).mean()
  slope = covariance.div(variance).item()
  intercept = (label_mean - slope * pred_mean).item()
  return slope, intercept


def apply_linear_regression_calibrator(preds, slope, intercept):
  """Apply a fitted affine regression calibrator to a list of predictions."""
  return [slope * pred + intercept for pred in preds]


def tune_binary_threshold(labels, scores):
  """Choose the binary decision threshold that maximizes F1 on calibration data."""
  best_threshold = 0.5
  best_score = -1.0

  for threshold in _candidate_binary_thresholds():
    preds = [1 if score >= threshold else 0 for score in scores]
    score = f1_score(labels, preds, zero_division=0)
    if score > best_score:
      best_score = score
      best_threshold = threshold

  return best_threshold


def apply_binary_threshold(scores, threshold):
  """Convert binary-class probabilities into hard labels with a fixed threshold."""
  return [1 if score >= threshold else 0 for score in scores]


def format_posthoc_classification_rows(predictions, task_metas):
  """Build the post-hoc binary-threshold tuning table rows for validation output."""
  rows = []

  for task_name in sorted(predictions):
    meta = task_metas[task_name]
    if meta["dtype"] != "bool":
      continue

    labels = predictions[task_name]["labels"]
    scores = predictions[task_name]["scores"]
    if len(labels) < 4:
      continue

    calib_indices, report_indices = _alternate_split_indices(len(labels))
    calib_labels = _select_items(labels, calib_indices)
    calib_scores = _select_items(scores, calib_indices)
    report_labels = _select_items(labels, report_indices)
    report_scores = _select_items(scores, report_indices)

    if len(set(calib_labels)) < 2 or len(set(report_labels)) < 2:
      continue

    threshold = tune_binary_threshold(calib_labels, calib_scores)
    report_preds = apply_binary_threshold(report_scores, threshold)
    report = _classification_report(report_labels, report_preds, report_scores, meta["dtype"])
    rows.append(
      [
        task_name,
        len(calib_labels),
        len(report_labels),
        _format_float(threshold),
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

  return rows


def format_posthoc_regression_rows(predictions, task_metas):
  """Build the post-hoc affine-regression calibration table rows for validation."""
  rows = []

  for task_name in sorted(predictions):
    meta = task_metas[task_name]
    if meta["dtype"] != "float":
      continue

    labels = predictions[task_name]["labels"]
    preds = predictions[task_name]["preds"]
    if len(labels) < 4:
      continue

    calib_indices, report_indices = _alternate_split_indices(len(labels))
    calib_labels = _select_items(labels, calib_indices)
    calib_preds = _select_items(preds, calib_indices)
    report_labels = _select_items(labels, report_indices)
    report_preds = _select_items(preds, report_indices)

    slope, intercept = fit_linear_regression_calibrator(calib_preds, calib_labels)
    calibrated_report_preds = apply_linear_regression_calibrator(report_preds, slope, intercept)
    report = _regression_report(report_labels, calibrated_report_preds)
    rows.append(
      [
        task_name,
        len(calib_labels),
        len(report_labels),
        _format_float(slope),
        _format_float(intercept),
        _format_float(report["pred_mean"]),
        _format_float(report["pred_std"]),
        _format_float(report["mae"]),
        _format_float(report["rmse"]),
        _format_float(report["spearman"]),
      ]
    )

  return rows


def fit_posthoc_calibration(predictions, task_metas, calibration_split="validation"):
  """Fit per-task post-hoc calibrators from one labeled prediction set.

  Regression tasks receive an affine calibrator `y = slope * pred + intercept`.
  Binary classification tasks receive an F1-tuned decision threshold. Multiclass
  tasks are left unchanged because this module currently only supports threshold
  tuning for binary outputs.
  """
  calibration = {
    "source_split": calibration_split,
    "classification": {},
    "regression": {},
  }

  for task_name in sorted(predictions):
    meta = task_metas[task_name]
    labels = predictions[task_name]["labels"]
    if not labels:
      continue

    if meta["dtype"] == "float":
      preds = predictions[task_name]["preds"]
      slope, intercept = fit_linear_regression_calibrator(preds, labels)
      calibration["regression"][task_name] = {
        "slope": slope,
        "intercept": intercept,
        "calibration_size": len(labels),
      }
    elif meta["dtype"] == "bool":
      scores = predictions[task_name]["scores"]
      if len(set(labels)) < 2:
        continue
      threshold = tune_binary_threshold(labels, scores)
      calibration["classification"][task_name] = {
        "threshold": threshold,
        "calibration_size": len(labels),
      }

  return calibration


def apply_posthoc_calibration(predictions, task_metas, calibration):
  """Apply saved per-task post-hoc calibrators to a prediction dictionary."""
  calibrated_predictions = {}
  classification_calibration = (calibration or {}).get("classification", {})
  regression_calibration = (calibration or {}).get("regression", {})

  for task_name, values in predictions.items():
    task_values = {
      key: list(value) if isinstance(value, list) else value
      for key, value in values.items()
    }
    meta = task_metas[task_name]

    if meta["dtype"] == "float" and task_name in regression_calibration:
      params = regression_calibration[task_name]
      task_values["preds"] = apply_linear_regression_calibrator(
        task_values["preds"],
        params["slope"],
        params["intercept"],
      )
    elif meta["dtype"] == "bool" and task_name in classification_calibration:
      params = classification_calibration[task_name]
      task_values["preds"] = apply_binary_threshold(task_values["scores"], params["threshold"])

    calibrated_predictions[task_name] = task_values

  return calibrated_predictions


def _format_float(value):
  if value is None:
    return "-"
  return f"{value:.4f}"


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
      report["auroc"] = roc_auc_score(labels, scores)
      report["auprc"] = average_precision_score(labels, scores)
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


def _alternate_split_indices(num_items):
  # Validation predictions are produced in a deterministic order that is influenced
  # by sequence-length batching. Splitting by "first half / second half" would make
  # the calibration subset and reporting subset systematically different.
  return list(range(0, num_items, 2)), list(range(1, num_items, 2))


def _select_items(values, indices):
  return [values[idx] for idx in indices]


def _candidate_binary_thresholds():
  return [step / 100.0 for step in range(5, 96)]
