"""Initialize MLflow experiment for the AML project.

Uses SQLite backend (filesystem backend is deprecated in MLflow 3.x).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlflow
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "mlflow.db"
ARTIFACTS_DIR = PROJECT_ROOT / "mlflow_artifacts"
EXPERIMENT_NAME = "aml-network-intelligence"

TRACKING_URI = f"sqlite:///{DB_PATH}"


def setup() -> str:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    logger.info(f"Tracking URI: {TRACKING_URI}")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=f"file://{ARTIFACTS_DIR}",
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
