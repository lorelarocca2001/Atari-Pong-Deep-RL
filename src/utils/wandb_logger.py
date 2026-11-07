
import wandb


class WandbLogger:
    """Manage Weights & Biases experiment tracking for DQN training."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict
    ):
        """
        Initialize a W&B experiment.

        Args:
            project: W&B project name.
            run_name: Name of the training run.
            config: Dictionary of training hyperparameters.
        """
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config
        )

        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")

    def log(self, metrics: dict, step: int) -> None:
        """
        Record training metrics at a given environment step.

        Args:
            metrics: Dictionary containing metric names and values.
            step: Current environment step.
        """
        self.run.log({
            "global_step": step,
            **metrics
        })

    def finish(self) -> None:
        """Finish the W&B experiment and synchronize logged metrics."""
        self.run.finish()
