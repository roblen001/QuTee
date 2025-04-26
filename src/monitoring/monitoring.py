"""Handles main class for training monitoring and saves plots to experimental_results."""
import os
import matplotlib.pyplot as plt

class TrainingMonitor:
    """
    Class for monitoring training metrics such as training loss, validation loss, and epoch duration.
    """

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.epoch_times: list[float] = []

        self.results_dir = "experimental_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def record(self, train_loss: float, val_loss: float, epoch_time: float) -> None:
        """
        Records the training and validation losses along with epoch duration.

        Args:
            train_loss (float): The training loss of the current epoch.
            val_loss (float): The validation loss of the current epoch.
            epoch_time (float): Duration of the current epoch in seconds.
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epoch_times.append(epoch_time)

    def plot_losses(self) -> None:
        """
        Plots the training and validation losses over epochs and saves the plot.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs (per 500 steps)')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.results_dir, "training_validation_loss.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved training/validation loss plot to {save_path}")

    def plot_epoch_times(self) -> None:
        """
        Plots the epoch durations over epochs and saves the plot.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_times, label='Epoch Time (seconds)', color='purple')
        plt.xlabel('Epochs (per 500 steps)')
        plt.ylabel('Time (seconds)')
        plt.title('Epoch Times Over Training')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.results_dir, "epoch_times.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved epoch times plot to {save_path}")
