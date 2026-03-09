"""
Aggregate protein-property datasets into a DuckDB database.

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
  # Dataset configuration for one property task.
  # `dtype` drives label coercion (all labels are stored as float in `samples`).
  # `head_type`, `num_classes`, and `loss` are training metadata for downstream loaders.
  task_name: str
  dataset: str
  dtype: str
  head_type: str
  num_classes: Optional[int]
  loss: str
  splits: Iterable[str] = ("train", "validation", "test")
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
  # ProtienGym DMS Substitution dataset
  TaskSpec(
    task_name="ProteinGym",
    dataset="ProteinGym",
    dtype="float",
    head_type="sequence_regression",
    num_classes=None,
    loss="mse",
    sequence_col="mutated_sequence",
    label_col="DMS_score",
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


def _loadcsv_dataset(
  path: Path,
  split_col: str = "split",
) -> Dict[str, object]:

  # Load all csvs
  # Assumes that the data has already been split into train/val/test
  # Creates a CSVSdataset wrapper to provide column names and iterable rows
  # Returns train/val/tet spilt as a dict

  dfs = [pd.read_csv(f) for f in path.glob("*.csv")]
  if not dfs:
    raise ValueError(f"No CSV files found in {path}")
  df = pd.concat(dfs, ignore_index=True)

  train_df = df[df[split_col] == "train"].reset_index(drop=True)
  val_df = df[df[split_col] == "validation"].reset_index(drop=True)
  test_df = df[df[split_col] == "test"].reset_index(drop=True)

  ds_dict = {"train": CSVDataset(train_df), "validation": CSVDataset(val_df), "test": CSVDataset(test_df)}
  return ds_dict


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


def _iter_selected_splits(task: TaskSpec, ds_dict: Dict[str, Any]) -> List[str]:
  # Keep configured split order while selecting only splits present in the dataset.
  # If none of the expected split names exist, process every split provided.
  available = set(ds_dict.keys())
  selected = [split for split in task.splits if split in available]
  if selected:
    return selected
  return list(ds_dict.keys())


def _source_name(task: TaskSpec, split: str) -> str:
  # Keep source as dataset/subset only (no split stored in DB).
  _ = split
  base = task.dataset if task.subset is None else f"{task.dataset}:{task.subset}"
  return base


def _insert_task_samples(
  con: duckdb.DuckDBPyConnection,
  task: TaskSpec,
  cache_dir: Optional[str],
  proteingym: Optional[str],
):
  # Load a task dataset and insert normalized sample rows.
  # We call `load_dataset` without split=... so we can iterate all available splits.
  if "proteingym" in task.dataset.lower():
    ds_dict = _loadcsv_dataset(Path(proteingym))
  else:
    ds_dict = load_dataset(task.dataset, task.subset, cache_dir=cache_dir)
  selected_splits = _iter_selected_splits(task, ds_dict)

  # Inserted count reflects rows accepted by DB uniqueness constraints.
  total_inserted = 0
  total_skipped = 0
  total_conflicts = 0

  for split in selected_splits:
    ds = ds_dict[split]
    # Resolve dataset-specific schema to our canonical sequence/label fields.
    sequence_col = _resolve_column(ds.column_names, task.sequence_col, SEQ_COL_CANDIDATES, "sequence", task.task_name)
    label_col = _resolve_column(ds.column_names, task.label_col, LABEL_COL_CANDIDATES, "label", task.task_name)
    source = _source_name(task, split)

    rows = []
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

      # Store labels as float regardless of task type.
      rows.append((seq, source, task.task_name, lbl))

      if len(rows) >= 5000:
        # First dataset wins: conflicts are skipped via DO NOTHING.
        # Row-wise RETURNING lets us count accepted vs conflicted rows.
        for row in rows:
          inserted = con.execute(
            """
            INSERT INTO samples(sequence, source, task_name, label)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sequence, task_name) DO NOTHING
            RETURNING 1
            """,
            row,
          ).fetchone()
          if inserted is None:
            total_conflicts += 1
          else:
            total_inserted += 1
        rows = []

    if rows:
      # Flush remaining rows after the final partial batch.
      for row in rows:
        inserted = con.execute(
          """
          INSERT INTO samples(sequence, source, task_name, label)
          VALUES (?, ?, ?, ?)
          ON CONFLICT(sequence, task_name) DO NOTHING
          RETURNING 1
          """,
          row,
        ).fetchone()
        if inserted is None:
          total_conflicts += 1
        else:
          total_inserted += 1

  print(f"Task={task.task_name} inserted={total_inserted} skipped_missing={total_skipped} skipped_conflict={total_conflicts}")


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
