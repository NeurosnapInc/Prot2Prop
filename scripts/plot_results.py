"""
Create publication-style Plotly figures from the metrics recorded in README.md.

The source-of-truth numbers here are transcribed directly from the README result
blocks so figure generation does not depend on re-running validation.
"""

from pathlib import Path

from plotly import graph_objects as go
from plotly.subplots import make_subplots


OUT_DIR = Path("figures/results")


ARCHITECTURE_VERSIONS = [
  {
    "version": "2026-04-26",
    "classification": {
      "material_production": {"f1": 0.8393, "auroc": 0.8463, "auprc": 0.9265},
      "solubility": {"f1": 0.7397, "auroc": 0.8619, "auprc": 0.8231},
      "temperature_stability": {"f1": 0.9234, "auroc": 0.9829, "auprc": 0.9832},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.8434, "rmse": 1.1004, "spearman": 0.7860},
      "expression_yield": {"mae": 0.6129, "rmse": 1.0296, "spearman": 0.6740},
      "folding_stability": {"mae": 0.7812, "rmse": 0.9900, "spearman": 0.8133},
    },
  },
  {
    "version": "2026-04-28",
    "classification": {
      "material_production": {"f1": 0.8349, "auroc": 0.8459, "auprc": 0.9259},
      "solubility": {"f1": 0.7380, "auroc": 0.8592, "auprc": 0.8190},
      "temperature_stability": {"f1": 0.9170, "auroc": 0.9837, "auprc": 0.9839},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.8551, "rmse": 1.1350, "spearman": 0.7781},
      "expression_yield": {"mae": 0.6418, "rmse": 1.0886, "spearman": 0.6688},
      "folding_stability": {"mae": 0.8535, "rmse": 1.0647, "spearman": 0.7964},
    },
  },
  {
    "version": "2026-04-29",
    "classification": {
      "material_production": {"f1": 0.8362, "auroc": 0.8458, "auprc": 0.9262},
      "solubility": {"f1": 0.7443, "auroc": 0.8640, "auprc": 0.8296},
      "temperature_stability": {"f1": 0.9233, "auroc": 0.9838, "auprc": 0.9842},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.7268, "rmse": 0.9547, "spearman": 0.8452},
      "expression_yield": {"mae": 0.5464, "rmse": 0.9304, "spearman": 0.7267},
      "folding_stability": {"mae": 0.6260, "rmse": 0.8305, "spearman": 0.8373},
    },
  },
  {
    "version": "2026-04-30",
    "classification": {
      "material_production": {"f1": 0.8359, "auroc": 0.8437, "auprc": 0.9247},
      "solubility": {"f1": 0.7424, "auroc": 0.8660, "auprc": 0.8300},
      "temperature_stability": {"f1": 0.9287, "auroc": 0.9840, "auprc": 0.9842},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.6876, "rmse": 0.9205, "spearman": 0.8522},
      "expression_yield": {"mae": 0.5564, "rmse": 0.9491, "spearman": 0.7117},
      "folding_stability": {"mae": 0.6883, "rmse": 0.8927, "spearman": 0.8322},
    },
  },
  {
    "version": "2026-05-22-evo",
    "classification": {
      "material_production": {"f1": 0.8338, "auroc": 0.8408, "auprc": 0.9222},
      "solubility": {"f1": 0.7455, "auroc": 0.8667, "auprc": 0.8315},
      "temperature_stability": {"f1": 0.9257, "auroc": 0.9836, "auprc": 0.9839},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.7264, "rmse": 0.9615, "spearman": 0.8550},
      "expression_yield": {"mae": 0.5473, "rmse": 0.9305, "spearman": 0.7289},
      "folding_stability": {"mae": 0.6435, "rmse": 0.8410, "spearman": 0.8440},
    },
  },
]


FINAL_RUNS = {
  "Seed 42": {
    "classification": {
      "material_production": {"f1": 0.8290, "auroc": 0.8424},
      "solubility": {"f1": 0.7398, "auroc": 0.8644},
      "temperature_stability": {"f1": 0.9243, "auroc": 0.9829},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.7569, "spearman": 0.8276},
      "expression_yield": {"mae": 0.5560, "spearman": 0.6973},
      "folding_stability": {"mae": 0.6824, "spearman": 0.8345},
    },
  },
  "Seed 26": {
    "classification": {
      "material_production": {"f1": 0.8360, "auroc": 0.8443},
      "solubility": {"f1": 0.7451, "auroc": 0.8652},
      "temperature_stability": {"f1": 0.9239, "auroc": 0.9827},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.7310, "spearman": 0.8516},
      "expression_yield": {"mae": 0.5551, "spearman": 0.6899},
      "folding_stability": {"mae": 0.6922, "spearman": 0.8300},
    },
  },
  "Seed 1": {
    "classification": {
      "material_production": {"f1": 0.8373, "auroc": 0.8445},
      "solubility": {"f1": 0.7456, "auroc": 0.8675},
      "temperature_stability": {"f1": 0.9290, "auroc": 0.9834},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.7238, "spearman": 0.8616},
      "expression_yield": {"mae": 0.5545, "spearman": 0.7336},
      "folding_stability": {"mae": 0.6795, "spearman": 0.8412},
    },
  },
  "Seed M": {
    "classification": {
      "material_production": {"f1": 0.8299, "auroc": 0.8399},
      "solubility": {"f1": 0.7444, "auroc": 0.8660},
      "temperature_stability": {"f1": 0.9242, "auroc": 0.9834},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.8071, "spearman": 0.8263},
      "expression_yield": {"mae": 0.5473, "spearman": 0.7222},
      "folding_stability": {"mae": 0.6400, "spearman": 0.8394},
    },
  },
  "Ensemble": {
    "classification": {
      "material_production": {"f1": 0.8360, "auroc": 0.8480},
      "solubility": {"f1": 0.7499, "auroc": 0.8714},
      "temperature_stability": {"f1": 0.9279, "auroc": 0.9840},
    },
    "regression": {
      "aggregation_propensity": {"mae": 0.7190, "spearman": 0.8575},
      "expression_yield": {"mae": 0.5498, "spearman": 0.7182},
      "folding_stability": {"mae": 0.6660, "spearman": 0.8432},
    },
  },
}


SEED1_CALIBRATION = {
  "classification_raw_f1": {
    "material_production": 0.8373,
    "solubility": 0.7456,
    "temperature_stability": 0.9290,
  },
  "classification_calibrated_f1": {
    "material_production": 0.8565,
    "solubility": 0.7474,
    "temperature_stability": 0.9323,
  },
  "regression_raw_mae": {
    "aggregation_propensity": 0.7238,
    "expression_yield": 0.5545,
    "folding_stability": 0.6795,
  },
  "regression_calibrated_mae": {
    "aggregation_propensity": 0.6809,
    "expression_yield": 0.5716,
    "folding_stability": 0.4884,
  },
  "regression_raw_rmse": {
    "aggregation_propensity": 0.9559,
    "expression_yield": 0.9377,
    "folding_stability": 0.8800,
  },
  "regression_calibrated_rmse": {
    "aggregation_propensity": 0.8950,
    "expression_yield": 0.8973,
    "folding_stability": 0.6411,
  },
}


NEUROSNAP_COLORS = {
  "primary": "#4361EE",
  "secondary": "#21b9dc",
  "tertiary": "#1bd666",
  "success": "#15c48d",
  "indigo": "#5b5bd6",
  "violet": "#7c4dff",
  "warn": "#e69801",
  "error": "#ff266a",
  "ink": "#0b3558",
}


TASK_COLORS = {
  "material_production": NEUROSNAP_COLORS["primary"],
  "solubility": NEUROSNAP_COLORS["violet"],
  "temperature_stability": NEUROSNAP_COLORS["tertiary"],
  "aggregation_propensity": NEUROSNAP_COLORS["success"],
  "expression_yield": NEUROSNAP_COLORS["indigo"],
  "folding_stability": NEUROSNAP_COLORS["secondary"],
}


def _apply_theme(fig, title):
  fig.update_layout(
    title=title,
    template="plotly_white",
    font={"family": "Arial, sans-serif", "size": 14},
    title_font={"size": 22},
    legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "center", "x": 0.5},
    margin={"l": 60, "r": 30, "t": 80, "b": 110},
  )
  return fig


def build_architecture_progression_figure():
  versions = [entry["version"] for entry in ARCHITECTURE_VERSIONS]
  fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
      "Classification F1",
      "Classification AUROC",
      "Regression MAE",
      "Regression Spearman",
    ),
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
  )

  for task in ("material_production", "solubility", "temperature_stability"):
    color = TASK_COLORS[task]
    fig.add_trace(
      go.Scatter(
        x=versions,
        y=[entry["classification"][task]["f1"] for entry in ARCHITECTURE_VERSIONS],
        mode="lines+markers",
        name=task,
        legendgroup=task,
        line={"color": color, "width": 3},
        marker={"size": 8},
      ),
      row=1,
      col=1,
    )
    fig.add_trace(
      go.Scatter(
        x=versions,
        y=[entry["classification"][task]["auroc"] for entry in ARCHITECTURE_VERSIONS],
        mode="lines+markers",
        name=task,
        legendgroup=task,
        showlegend=False,
        line={"color": color, "width": 3, "dash": "dot"},
        marker={"size": 8},
      ),
      row=1,
      col=2,
    )

  for task in ("aggregation_propensity", "expression_yield", "folding_stability"):
    color = TASK_COLORS[task]
    fig.add_trace(
      go.Scatter(
        x=versions,
        y=[entry["regression"][task]["mae"] for entry in ARCHITECTURE_VERSIONS],
        mode="lines+markers",
        name=task,
        legendgroup=task,
        line={"color": color, "width": 3},
        marker={"size": 8},
      ),
      row=2,
      col=1,
    )
    fig.add_trace(
      go.Scatter(
        x=versions,
        y=[entry["regression"][task]["spearman"] for entry in ARCHITECTURE_VERSIONS],
        mode="lines+markers",
        name=task,
        legendgroup=task,
        showlegend=False,
        line={"color": color, "width": 3, "dash": "dot"},
        marker={"size": 8},
      ),
      row=2,
      col=2,
    )

  fig.update_yaxes(title_text="F1", row=1, col=1)
  fig.update_yaxes(title_text="AUROC", row=1, col=2)
  fig.update_yaxes(title_text="MAE", row=2, col=1)
  fig.update_yaxes(title_text="Spearman", row=2, col=2)
  fig.update_xaxes(tickangle=-35)
  return _apply_theme(fig, "Prot2Prop Architecture Progression Across Development Versions")


def build_seed_comparison_figure():
  runs = list(FINAL_RUNS.keys())
  fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
      "Classification F1",
      "Classification AUROC",
      "Regression MAE",
      "Regression Spearman",
    ),
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
  )

  for task in ("material_production", "solubility", "temperature_stability"):
    color = TASK_COLORS[task]
    fig.add_trace(
      go.Bar(
        x=runs,
        y=[FINAL_RUNS[run]["classification"][task]["f1"] for run in runs],
        name=task,
        legendgroup=task,
        marker_color=color,
      ),
      row=1,
      col=1,
    )
    fig.add_trace(
      go.Bar(
        x=runs,
        y=[FINAL_RUNS[run]["classification"][task]["auroc"] for run in runs],
        name=task,
        legendgroup=task,
        showlegend=False,
        marker_color=color,
      ),
      row=1,
      col=2,
    )

  for task in ("aggregation_propensity", "expression_yield", "folding_stability"):
    color = TASK_COLORS[task]
    fig.add_trace(
      go.Bar(
        x=runs,
        y=[FINAL_RUNS[run]["regression"][task]["mae"] for run in runs],
        name=task,
        legendgroup=task,
        marker_color=color,
      ),
      row=2,
      col=1,
    )
    fig.add_trace(
      go.Bar(
        x=runs,
        y=[FINAL_RUNS[run]["regression"][task]["spearman"] for run in runs],
        name=task,
        legendgroup=task,
        showlegend=False,
        marker_color=color,
      ),
      row=2,
      col=2,
    )

  fig.update_layout(barmode="group")
  fig.update_yaxes(title_text="F1", row=1, col=1)
  fig.update_yaxes(title_text="AUROC", row=1, col=2)
  fig.update_yaxes(title_text="MAE", row=2, col=1)
  fig.update_yaxes(title_text="Spearman", row=2, col=2)
  fig.update_xaxes(tickangle=-20)
  return _apply_theme(fig, "Final Architecture Comparison Across Seeds and Ensemble")


def build_seed1_calibration_figure():
  fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
      "Classification F1",
      "Regression MAE",
      "Regression RMSE",
    ),
    horizontal_spacing=0.12,
  )

  class_tasks = list(SEED1_CALIBRATION["classification_raw_f1"].keys())
  reg_tasks = list(SEED1_CALIBRATION["regression_raw_mae"].keys())

  fig.add_trace(
    go.Bar(
      x=class_tasks,
      y=[SEED1_CALIBRATION["classification_raw_f1"][task] for task in class_tasks],
      name="Raw",
      marker_color=NEUROSNAP_COLORS["ink"],
    ),
    row=1,
    col=1,
  )
  fig.add_trace(
    go.Bar(
      x=class_tasks,
      y=[SEED1_CALIBRATION["classification_calibrated_f1"][task] for task in class_tasks],
      name="Calibrated",
      marker_color=NEUROSNAP_COLORS["success"],
    ),
    row=1,
    col=1,
  )
  fig.add_trace(
    go.Bar(
      x=reg_tasks,
      y=[SEED1_CALIBRATION["regression_raw_mae"][task] for task in reg_tasks],
      name="Raw",
      showlegend=False,
      marker_color=NEUROSNAP_COLORS["ink"],
    ),
    row=1,
    col=2,
  )
  fig.add_trace(
    go.Bar(
      x=reg_tasks,
      y=[SEED1_CALIBRATION["regression_calibrated_mae"][task] for task in reg_tasks],
      name="Calibrated",
      showlegend=False,
      marker_color=NEUROSNAP_COLORS["success"],
    ),
    row=1,
    col=2,
  )
  fig.add_trace(
    go.Bar(
      x=reg_tasks,
      y=[SEED1_CALIBRATION["regression_raw_rmse"][task] for task in reg_tasks],
      name="Raw",
      showlegend=False,
      marker_color=NEUROSNAP_COLORS["ink"],
    ),
    row=1,
    col=3,
  )
  fig.add_trace(
    go.Bar(
      x=reg_tasks,
      y=[SEED1_CALIBRATION["regression_calibrated_rmse"][task] for task in reg_tasks],
      name="Calibrated",
      showlegend=False,
      marker_color=NEUROSNAP_COLORS["success"],
    ),
    row=1,
    col=3,
  )

  fig.update_layout(barmode="group")
  fig.update_yaxes(title_text="F1", row=1, col=1)
  fig.update_yaxes(title_text="MAE", row=1, col=2)
  fig.update_yaxes(title_text="RMSE", row=1, col=3)
  fig.update_xaxes(tickangle=-30)
  return _apply_theme(fig, "Effect of Post-hoc Calibration on the Final Seed 1 Checkpoint")


def build_best_single_vs_ensemble_figure():
  reference = "Seed 1"
  compare = "Ensemble"
  fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Classification Tasks", "Regression Tasks"),
    horizontal_spacing=0.15,
  )

  class_tasks = ["material_production", "solubility", "temperature_stability"]
  reg_tasks = ["aggregation_propensity", "expression_yield", "folding_stability"]

  fig.add_trace(
    go.Bar(
      x=class_tasks,
      y=[FINAL_RUNS[reference]["classification"][task]["f1"] for task in class_tasks],
      name="Seed 1",
      marker_color=NEUROSNAP_COLORS["primary"],
    ),
    row=1,
    col=1,
  )
  fig.add_trace(
    go.Bar(
      x=class_tasks,
      y=[FINAL_RUNS[compare]["classification"][task]["f1"] for task in class_tasks],
      name="Ensemble",
      marker_color=NEUROSNAP_COLORS["secondary"],
    ),
    row=1,
    col=1,
  )
  fig.add_trace(
    go.Bar(
      x=reg_tasks,
      y=[FINAL_RUNS[reference]["regression"][task]["mae"] for task in reg_tasks],
      name="Seed 1 MAE",
      marker_color=NEUROSNAP_COLORS["primary"],
      showlegend=False,
    ),
    row=1,
    col=2,
  )
  fig.add_trace(
    go.Bar(
      x=reg_tasks,
      y=[FINAL_RUNS[compare]["regression"][task]["mae"] for task in reg_tasks],
      name="Ensemble MAE",
      marker_color=NEUROSNAP_COLORS["secondary"],
      showlegend=False,
    ),
    row=1,
    col=2,
  )

  fig.update_layout(barmode="group")
  fig.update_yaxes(title_text="F1 (higher is better)", row=1, col=1)
  fig.update_yaxes(title_text="MAE (lower is better)", row=1, col=2)
  fig.update_xaxes(tickangle=-30)
  return _apply_theme(fig, "Best Single Checkpoint Versus Four-checkpoint Ensemble")


def write_figure(fig, filename):
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  fig.write_html(OUT_DIR / f"{filename}.html", include_plotlyjs="cdn")
  try:
    fig.write_image(OUT_DIR / f"{filename}.png", scale=2, width=1600, height=900)
  except Exception:
    pass


def main():
  figures = {
    "architecture_progression": build_architecture_progression_figure(),
    "final_seed_comparison": build_seed_comparison_figure(),
    "seed1_calibration": build_seed1_calibration_figure(),
    "seed1_vs_ensemble": build_best_single_vs_ensemble_figure(),
  }
  for name, fig in figures.items():
    write_figure(fig, name)
  print(f"Wrote {len(figures)} figures to {OUT_DIR}")


if __name__ == "__main__":
  main()
