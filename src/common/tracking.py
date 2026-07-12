"""MLflow experiment tracking (local file store; no login/network).

Replaces the previous Weights & Biases integration: file-based mlruns/, a local
`mlflow ui` dashboard, and identical logging for classical and quantum so sweeps
compare cleanly.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import mlflow

# Enable file store backend (required in MLflow 3.x+)
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class _RunLogger:
    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_figure(self, fig, filename: str) -> None:
        mlflow.log_figure(fig, filename)


@contextmanager
def run(experiment: str, run_name: str, params: dict, tracking_uri: str | None = None):
    """Open one MLflow run, log params, yield a logger, and close on exit.

    tracking_uri defaults to a local ./mlruns file store when None.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        yield _RunLogger()
