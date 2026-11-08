
from pathlib import Path

import wandb


class WandbLogger:
    """Manage Weights & Biases tracking for DQN training."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict,
        log_dir: str = "experiments/logs",
        resume: bool = False
    ):
        """
        Initialize or resume a W&B experiment.

        Args:
            project: W&B project name.
            run_name: Name of the experiment.
            config: Training hyperparameters.
            log_dir: Directory for local W&B files.
            resume: Whether to resume an existing W&B run.
        """
        log_path = Path(log_dir).resolve()
        log_path.mkdir(parents=True, exist_ok=True)

        run_id_path = log_path / "wandb_run_id.txt"

        if resume:
            if not run_id_path.is_file():
                raise FileNotFoundError(
                    f"W&B run ID not found: {run_id_path}"
                )

            run_id = run_id_path.read_text(
                encoding="utf-8"
            ).strip()

            if not run_id:
                raise ValueError("W&B run ID is empty.")
        else:
            if run_id_path.exists():
                raise FileExistsError(
                    f"W&B run ID already exists: {run_id_path}"
                )

            run_id = None

        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            dir=str(log_path),
            id=run_id,
            resume="must" if resume else "never"
        )

        if not resume:
            run_id_path.write_text(
                self.run.id,
                encoding="utf-8"
            )

        self.run.define_metric("global_step")
        self.run.define_metric(
            "*",
            step_metric="global_step"
        )

    def log(self, metrics: dict, step: int) -> None:
        """
        Record training metrics at an environment step.

        Args:
            metrics: Dictionary of metric names and values.
            step: Current environment step.
        """
        self.run.log({
            "global_step": step,
            **metrics
        })

    def finish(self) -> None:
        """Finish and synchronize the W&B experiment."""
        self.run.finish()


