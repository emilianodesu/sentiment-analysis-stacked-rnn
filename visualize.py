"""Visualization utilities for training curves, confusion matrices, and gradients."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from evaluate import EvaluationResults
from train import TrainingHistory


def plot_training_curves(
    lstm_history: TrainingHistory, gru_history: TrainingHistory
) -> Figure:
    """Create training curves comparing LSTM and GRU.

    Generates a figure with 2 subplots:
    - Left: Training and validation loss for both models
    - Right: Training and validation accuracy for both models

    Args:
        lstm_history: Training history from the LSTM model.
        gru_history: Training history from the GRU model.

    Returns:
        Matplotlib Figure with the training curves.
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    # Epoch indices
    lstm_epochs = range(1, len(lstm_history.train_losses) + 1)
    gru_epochs = range(1, len(gru_history.train_losses) + 1)

    # Loss subplot
    ax_loss.plot(lstm_epochs, lstm_history.train_losses, "b-", label="LSTM Train")
    ax_loss.plot(lstm_epochs, lstm_history.val_losses, "b--", label="LSTM Val")
    ax_loss.plot(gru_epochs, gru_history.train_losses, "r-", label="GRU Train")
    ax_loss.plot(gru_epochs, gru_history.val_losses, "r--", label="GRU Val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.legend()

    # Accuracy subplot
    ax_acc.plot(lstm_epochs, lstm_history.train_accs, "b-", label="LSTM Train")
    ax_acc.plot(lstm_epochs, lstm_history.val_accs, "b--", label="LSTM Val")
    ax_acc.plot(gru_epochs, gru_history.train_accs, "r-", label="GRU Train")
    ax_acc.plot(gru_epochs, gru_history.val_accs, "r--", label="GRU Val")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training and Validation Accuracy")
    ax_acc.legend()

    fig.suptitle("LSTM vs GRU Training Curves", fontsize=14)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    lstm_results: EvaluationResults, gru_results: EvaluationResults
) -> Figure:
    """Create side-by-side confusion matrix heatmaps.

    Generates a figure with 2 subplots showing annotated heatmaps
    for LSTM and GRU confusion matrices.

    Args:
        lstm_results: Evaluation results from the LSTM model.
        gru_results: Evaluation results from the GRU model.

    Returns:
        Matplotlib Figure with the confusion matrix heatmaps.
    """
    fig, (ax_lstm, ax_gru) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Negative", "Positive"]

    # LSTM confusion matrix
    sns.heatmap(
        np.array(lstm_results.confusion_matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_lstm,
    )
    ax_lstm.set_xlabel("Predicted")
    ax_lstm.set_ylabel("Actual")
    ax_lstm.set_title(f"LSTM Confusion Matrix")

    # GRU confusion matrix
    sns.heatmap(
        np.array(gru_results.confusion_matrix),
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_gru,
    )
    ax_gru.set_xlabel("Predicted")
    ax_gru.set_ylabel("Actual")
    ax_gru.set_title(f"GRU Confusion Matrix")

    fig.suptitle("Confusion Matrices: LSTM vs GRU", fontsize=14)
    fig.tight_layout()
    return fig


def plot_gradient_norms(
    lstm_history: TrainingHistory, gru_history: TrainingHistory
) -> Figure:
    """Create gradient L2 norm plots per layer for steps 1-100.

    Generates a figure with 2 subplots:
    - Left: LSTM gradient norms per layer
    - Right: GRU gradient norms per layer

    Each layer gets its own line with a legend entry.

    Args:
        lstm_history: Training history with gradient norms from LSTM.
        gru_history: Training history with gradient norms from GRU.

    Returns:
        Matplotlib Figure with the gradient norm plots.
    """
    fig, (ax_lstm, ax_gru) = plt.subplots(1, 2, figsize=(14, 5))

    # LSTM gradient norms
    for layer_name, norms in lstm_history.gradient_norms.items():
        steps = range(1, len(norms) + 1)
        # Use a shorter label for readability
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        ax_lstm.plot(steps, norms, label=short_name)
    ax_lstm.set_xlabel("Step")
    ax_lstm.set_ylabel("Gradient L2 Norm")
    ax_lstm.set_title("LSTM Gradient Norms")
    ax_lstm.legend(fontsize="small", loc="upper right")

    # GRU gradient norms
    for layer_name, norms in gru_history.gradient_norms.items():
        steps = range(1, len(norms) + 1)
        short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
        ax_gru.plot(steps, norms, label=short_name)
    ax_gru.set_xlabel("Step")
    ax_gru.set_ylabel("Gradient L2 Norm")
    ax_gru.set_title("GRU Gradient Norms")
    ax_gru.legend(fontsize="small", loc="upper right")

    fig.suptitle("Gradient L2 Norms per Layer (Steps 1-100)", fontsize=14)
    fig.tight_layout()
    return fig


def save_all_plots(
    lstm_history: TrainingHistory,
    gru_history: TrainingHistory,
    lstm_results: EvaluationResults,
    gru_results: EvaluationResults,
    output_dir: str = "outputs",
) -> None:
    """Generate and save all visualization plots as PNG files.

    Calls each plot function, saves the resulting figures to the output
    directory, and closes them to free memory.

    Args:
        lstm_history: Training history from the LSTM model.
        gru_history: Training history from the GRU model.
        lstm_results: Evaluation results from the LSTM model.
        gru_results: Evaluation results from the GRU model.
        output_dir: Directory to save PNG files. Defaults to "outputs".
    """
    os.makedirs(output_dir, exist_ok=True)

    # Training curves
    fig_curves = plot_training_curves(lstm_history, gru_history)
    fig_curves.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig_curves)

    # Confusion matrices
    fig_cm = plot_confusion_matrices(lstm_results, gru_results)
    fig_cm.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=150)
    plt.close(fig_cm)

    # Gradient norms
    fig_grad = plot_gradient_norms(lstm_history, gru_history)
    fig_grad.savefig(os.path.join(output_dir, "gradient_norms.png"), dpi=150)
    plt.close(fig_grad)
