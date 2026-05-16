"""Model evaluation and metrics computation."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


@dataclass
class EvaluationResults:
    """Stores evaluation metrics for a trained sentiment model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    rnn_type: str


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str,
    rnn_type: str,
) -> EvaluationResults:
    """Evaluate a trained sentiment model on the test set.

    Loads the best checkpoint, sets the model to eval mode, runs inference
    with no_grad, applies sigmoid > 0.5 for binary predictions, and computes
    metrics using sklearn.

    Args:
        model: The SentimentRNN model instance (architecture must match checkpoint).
        test_loader: DataLoader for the test set.
        device: Device to run inference on (cuda, mps, or cpu).
        checkpoint_path: Path to the saved model checkpoint (.pt file).
        rnn_type: Type of RNN ("LSTM" or "GRU") for labeling results.

    Returns:
        EvaluationResults with accuracy, precision, recall, f1, confusion matrix,
        and rnn_type.
    """
    # Load checkpoint weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Move model to device and set eval mode
    model = model.to(device)
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []

    # Inference with no gradient computation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: model outputs logits of shape (batch_size, 1)
            logits = model(inputs).squeeze(1)

            # Apply sigmoid and threshold at 0.5
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    # Convert to numpy arrays for sklearn
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvaluationResults(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm,
        rnn_type=rnn_type,
    )


def print_comparison_table(
    lstm_results: EvaluationResults, gru_results: EvaluationResults
) -> None:
    """Print a formatted comparison table showing LSTM vs GRU evaluation metrics.

    Displays accuracy, precision, recall, and F1 for both models side by side,
    with a difference column showing the delta between LSTM and GRU.

    Args:
        lstm_results: EvaluationResults from the LSTM model evaluation.
        gru_results: EvaluationResults from the GRU model evaluation.
    """
    separator = "=" * 60
    dash_separator = "-" * 60

    print(separator)
    print(f"{'LSTM vs GRU Comparison':^60}")
    print(separator)
    print(f"{'Metric':<16}|{'LSTM':^12}|{'GRU':^12}|{'Difference':^12}")
    print(dash_separator)

    metrics = [
        ("Accuracy", lstm_results.accuracy, gru_results.accuracy),
        ("Precision", lstm_results.precision, gru_results.precision),
        ("Recall", lstm_results.recall, gru_results.recall),
        ("F1 Score", lstm_results.f1, gru_results.f1),
    ]

    for name, lstm_val, gru_val in metrics:
        diff = lstm_val - gru_val
        sign = "+" if diff >= 0 else ""
        lstm_str = f"{lstm_val * 100:.2f}%"
        gru_str = f"{gru_val * 100:.2f}%"
        diff_str = f"{sign}{diff * 100:.2f}%"
        print(f"{name:<16}|{lstm_str:^12}|{gru_str:^12}|{diff_str:^12}")

    print(separator)
