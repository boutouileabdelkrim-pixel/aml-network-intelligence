"""Initialize MLflow experiment for the AML project.

Run once: creates the experiment + registers the tracking URI.
Subsequent runs (log_training) write to the same experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "aml-network-intelligence"


def setup() -> str:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"file://{MLRUNS_DIR}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Tracking URI: {tracking_uri}")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=f"{tracking_uri}/artifacts",
        )
        logger.success(f"Created experiment '{EXPERIMENT_NAME}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Experiment already exists (id={experiment_id})")

    return experiment_id


def main() -> None:
    exp_id = setup()
    print(f"EXPERIMENT_ID={exp_id}")


if __name__ == "__main__":
    sys.exit(main())
