"""
Aggregate protein-property datasets into a DuckDB database.
Inspect Results: duckdb -ui data/aggregated/aggregated.duckdb

Tables:
- samples(sequence, source, task_name, label)
  - UNIQUE(sequence, task_name)
- tasks(task_name, dtype, head_type, num_classes, loss)
  - PRIMARY KEY(task_name)
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class TaskSpec:
  """Dataset configuration for one prediction task.

  Args:
    task_name: Name of the prediction target. This should usually map directly
      to the biochemical property being predicted (for example
      `enzyme_activity`, `thermostability`, or `solubility`). Use non-property
      names only when there is a clear exception.
    dataset: Dataset identifier passed to `datasets.load_dataset`, or a marker
      name used by custom loaders (for example `ProteinGym` for local CSV
      loading in this script).
    dtype: Label type for coercion and downstream interpretation. Supported
      values in this script are `bool`, `int`, and `float`.
    head_type: Model-head family expected by downstream training code (for
      example `sequence_binary` or `sequence_regression`).
    num_classes: Number of target classes for classification tasks. Set to
      `None` for regression tasks.
    loss: Preferred loss name for downstream training metadata (for example
      `bce` or `mse`).
    sequence_col: Optional explicit sequence column name. If `None`, the script
      infers from known sequence column candidates.
    label_col: Optional explicit label column name. If `None`, the script
      infers from known label column candidates.
    subset: Optional dataset subset/config name passed as the second argument
      to `datasets.load_dataset`.
  """
  task_name: str
  dataset: str
  dtype: str
  head_type: str
  num_classes: Optional[int]
  loss: str
  sequence_col: Optional[str] = None
  label_col: Optional[str] = None
  subset: Optional[str] = None


class CSVDataset:
  # Class to load CSV datasets.
  # Provides `column_names` and iterable row dicts for aggregation scripts.
  def __init__(self, df: pd.DataFrame):
    self.rows = df.to_dict(orient="records")
    self.column_names = df.columns.tolist()

  def __iter__(self) -> Iterable[dict]:
    return iter(self.rows)

  def __len__(self):
    return len(self.rows)


# Source priority is defined by list order.
# Earlier entries are considered higher quality and are inserted first.
# Later entries cannot overwrite existing (sequence, task_name) rows.
TASKS: List[TaskSpec] = [
  # Material production as sequence-level binary classification.
  TaskSpec(
    task_name="material_production",
    dataset="AI4Protein/material_production",
    dtype="bool",
    head_type="sequence_binary",
    num_classes=2,
    loss="bce",
  ),
  # DeepSol has a known non-default sequence column (`aa_seq`).
  TaskSpec(
    task_name="solubility",
    dataset="AI4Protein/DeepSol",
    dtype="bool",
    head_type="sequence_binary",
    num_classes=2,
    loss="bce",
    sequence_col="aa_seq",
    label_col="label",
  ),
  # Temperature stability modeled as sequence-level regression.
  TaskSpec(
    task_name="temperature_stability",
    dataset="AI4Protein/temperature_stability",
    dtype="float",
    head_type="sequence_regression",
    num_classes=None,
    loss="mse",
  ),
  # ProteinGym DMS Substitution dataset
  TaskSpec(
    task_name='aggregation_propensity',
    dataset='ProteinGym/aggregation_propensity',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
  TaskSpec(
    task_name='binding_affinity',
    dataset='ProteinGym/binding_affinity',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
  TaskSpec(
    task_name='enzymatic_activity',
    dataset='ProteinGym/enzymatic_activity',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
  TaskSpec(
    task_name='expression_yield',
    dataset='ProteinGym/expression_yield',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
  TaskSpec(
    task_name='folding_stability',
    dataset='ProteinGym/folding_stability',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
  TaskSpec(
    task_name='membrane_topology',
    dataset='ProteinGym/membrane_topology',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
  TaskSpec(
    task_name='thermostability',
    dataset='ProteinGym/thermostability',
    dtype='float',
    head_type='sequence_regression',
    num_classes=None,
    loss='mse',
    sequence_col='mutated_sequence',
    label_col='DMS_score',
  ),
]


SEQ_COL_CANDIDATES = (
  "sequence",
  "aa_seq",
  "protein_sequence",
  "seq",
)

LABEL_COL_CANDIDATES = (
  "label",
  "target",
  "y",
  "value",
)

PROTEINGYM_MANIFEST_DIR = Path(__file__).resolve().parent / "proteingym_manifests"


def _manifest_path(task: TaskSpec) -> Path:
  return PROTEINGYM_MANIFEST_DIR / f"{task.task_name}.csv"


def _load_proteingym_dataset(
  task: TaskSpec,
  path: Path,
  indels_path: Optional[Path],
) -> Dict[str, object]:
  manifest_path = _manifest_path(task)
  if not manifest_path.exists():
    raise ValueError(f"ProteinGym manifest not found for task '{task.task_name}': {manifest_path}")

  manifest_df = pd.read_csv(manifest_path)
  if "DMS_filename" not in manifest_df.columns:
    raise KeyError(f"Manifest '{manifest_path}' is missing required column 'DMS_filename'")

  datasets: Dict[str, object] = {}
  missing_files: List[str] = []

  for filename in manifest_df["DMS_filename"].astype(str):
    csv_path = path / filename
    if not csv_path.exists():
      if indels_path is not None:
        fallback_path = indels_path / filename
        if fallback_path.exists():
          csv_path = fallback_path
        else:
          missing_files.append(filename)
          continue
      else:
        missing_files.append(filename)
        continue
    datasets[filename] = CSVDataset(pd.read_csv(csv_path))

  if not datasets:
    raise ValueError(
      f"No ProteinGym CSV files matched manifest '{manifest_path}' under {path}"
    )

  if missing_files:
    print(
      f"Task={task.task_name} manifest_missing_files={len(missing_files)} "
      f"(not found under {path}, likely absent from this ProteinGym subset)"
    )

  return datasets


def _resolve_column(column_names: List[str], preferred: Optional[str], candidates: Iterable[str], kind: str, task_name: str) -> str:
  # Pick a preferred column, else the first matching candidate.
  # This keeps task definitions short while still handling common schema variants.
  if preferred is not None:
    if preferred not in column_names:
      raise KeyError(f"Task '{task_name}' expected {kind} column '{preferred}', but columns are: {column_names}")
    return preferred

  for candidate in candidates:
    if candidate in column_names:
      return candidate

  raise KeyError(f"Task '{task_name}' could not infer a {kind} column from columns: {column_names}")


def _coerce_label(value: Any, dtype: str) -> Optional[float]:
  # Convert all labels to float; skip missing/empty labels.
  # Downstream training can cast back to bool/int based on `tasks.dtype`.
  if value is None:
    return None

  if isinstance(value, str):
    stripped = value.strip()
    if stripped == "":
      return None
    if dtype == "bool":
      lowered = stripped.lower()
      if lowered in ("true", "t", "yes", "y", "1", "positive", "pos"):
        return 1.0
      if lowered in ("false", "f", "no", "n", "0", "negative", "neg"):
        return 0.0
      return float(stripped)
    return float(stripped)

  if dtype == "bool":
    if isinstance(value, bool):
      return 1.0 if value else 0.0
    return 1.0 if float(value) > 0 else 0.0

  if dtype == "int":
    return float(int(value))

  return float(value)


def _prepare_db(con: duckdb.DuckDBPyConnection):
  # Create target tables with required constraints.
  # The script intentionally recreates tables from scratch on each run.
  con.execute("DROP TABLE IF EXISTS samples")
  con.execute("DROP TABLE IF EXISTS tasks")

  con.execute(
    """
    CREATE TABLE tasks (
      task_name VARCHAR PRIMARY KEY,
      dtype VARCHAR NOT NULL,
      head_type VARCHAR NOT NULL,
      num_classes INTEGER,
      loss VARCHAR NOT NULL
    )
    """
  )

  con.execute(
    """
    CREATE TABLE samples (
      sequence VARCHAR NOT NULL,
      source VARCHAR NOT NULL,
      task_name VARCHAR NOT NULL,
      label DOUBLE NOT NULL,
      -- Uniqueness Constraints: one label per (sequence, task_name).
      CONSTRAINT samples_sequence_task_unique UNIQUE(sequence, task_name),
      FOREIGN KEY (task_name) REFERENCES tasks(task_name)
    )
    """
  )


def _insert_task(con: duckdb.DuckDBPyConnection, task: TaskSpec):
  # One metadata row per task head.
  # If the same task_name appears multiple times, keep the first metadata row.
  con.execute(
    """
    INSERT INTO tasks(task_name, dtype, head_type, num_classes, loss)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(task_name) DO NOTHING
    """,
    [task.task_name, task.dtype, task.head_type, task.num_classes, task.loss],
  )

  # Ensure all sources that map to one task_name agree on metadata.
  row = con.execute(
    """
    SELECT dtype, head_type, num_classes, loss
    FROM tasks
    WHERE task_name = ?
    """,
    [task.task_name],
  ).fetchone()
  if row != (task.dtype, task.head_type, task.num_classes, task.loss):
    raise ValueError(
      f"Inconsistent metadata for task '{task.task_name}'. Existing={row}, incoming={(task.dtype, task.head_type, task.num_classes, task.loss)}"
    )


def _source_name(task: TaskSpec, dataset_key: str) -> str:
  # Keep source as dataset/subset only for HF datasets.
  base = task.dataset if task.subset is None else f"{task.dataset}:{task.subset}"
  if "proteingym" in task.dataset.lower():
    return f"{base}:{Path(dataset_key).stem}"
  return base


def _insert_task_samples(
  con: duckdb.DuckDBPyConnection,
  task: TaskSpec,
  cache_dir: Optional[str],
  proteingym: Optional[str],
):
  # Load a task dataset and insert normalized sample rows.
  # We load the full dataset object so aggregation can process every partition returned.
  if "proteingym" in task.dataset.lower():
    if proteingym is None:
      raise ValueError("ProteinGym path is required for ProteinGym tasks")
    proteingym_path = Path(proteingym)
    indels_path = proteingym_path.parent / "DMS_ProteinGym_indels"
    if indels_path == proteingym_path or not indels_path.exists():
      indels_path = None
    ds_dict = _load_proteingym_dataset(task, proteingym_path, indels_path)
  else:
    ds_dict = load_dataset(task.dataset, task.subset, cache_dir=cache_dir)

  total_skipped = 0
  total_duplicates = 0
  seen_sequences = set()
  rows = []

  for dataset_key, ds in ds_dict.items():
    # Resolve dataset-specific schema to our canonical sequence/label fields.
    sequence_col = _resolve_column(ds.column_names, task.sequence_col, SEQ_COL_CANDIDATES, "sequence", task.task_name)
    label_col = _resolve_column(ds.column_names, task.label_col, LABEL_COL_CANDIDATES, "label", task.task_name)
    source = _source_name(task, dataset_key)

    for ex in ds:
      seq = ex.get(sequence_col)
      if seq is None:
        total_skipped += 1
        continue

      seq = str(seq).strip()
      if seq == "":
        total_skipped += 1
        continue

      lbl = _coerce_label(ex.get(label_col), task.dtype)
      if lbl is None:
        total_skipped += 1
        continue

      if seq in seen_sequences:
        total_duplicates += 1
        continue

      seen_sequences.add(seq)

      # Store labels as float regardless of task type.
      rows.append((seq, source, task.task_name, lbl))

  if rows:
    task_df = pd.DataFrame(rows, columns=["sequence", "source", "task_name", "label"])
    con.register("task_rows", task_df)
    try:
      con.execute(
        """
        INSERT INTO samples(sequence, source, task_name, label)
        SELECT sequence, source, task_name, label
        FROM task_rows
        """
      )
    finally:
      con.unregister("task_rows")

  print(f"Task={task.task_name} inserted={len(rows)} skipped_missing={total_skipped} skipped_duplicate={total_duplicates}")


def aggregate(tasks: List[TaskSpec], out_db: Path, cache_dir: Optional[str], proteingym: Optional[str]):
  # Build the DuckDB file for all configured tasks.
  # The output DB is self-contained and can be queried directly via DuckDB/SQLite-style SQL workflows.
  out_db.parent.mkdir(parents=True, exist_ok=True)
  con = duckdb.connect(out_db.as_posix())
  try:
    _prepare_db(con)

    for task in tasks:
      _insert_task(con, task)
      _insert_task_samples(con, task, cache_dir, proteingym)

    total = con.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    print(f"Aggregation complete: {total} sample rows written to {out_db}")
  finally:
    con.close()


def _parse_args():
  # CLI arguments to set output and cache paths.
  parser = argparse.ArgumentParser(description="Aggregate multiple datasets into DuckDB tables.")
  parser.add_argument("--out-db", default="data/aggregated/aggregated.duckdb", help="Output DuckDB file path.")
  parser.add_argument("--cache-dir", default=None, help="Optional HuggingFace datasets cache directory.")
  parser.add_argument("--proteingym", default="DMS_ProteinGym_substitutions", help="Path to proteingym data")
  return parser.parse_args()


def main():
  # Entrypoint for CLI usage.
  args = _parse_args()
  aggregate(TASKS, Path(args.out_db), args.cache_dir, args.proteingym)


if __name__ == "__main__":
  main()
