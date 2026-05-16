"""Training loop with early stopping and gradient monitoring."""

import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    GRAD_CLIP_NORM,
    GRADIENT_MONITOR_STEPS,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Fixes seeds for Python's random, NumPy, PyTorch CPU and CUDA,
    and configures cuDNN for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainingHistory:
    """Stores training metrics across epochs.

    Attributes:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accs: Training accuracy per epoch.
        val_accs: Validation accuracy per epoch.
        gradient_norms: Gradient L2 norms per layer name for steps 1-100.
        training_time: Total training time in seconds.
        epochs_trained: Number of epochs completed (may be < MAX_EPOCHS with early stopping).
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    gradient_norms: dict[str, list[float]] = field(default_factory=dict)
    training_time: float = 0.0
    epochs_trained: int = 0


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = MAX_EPOCHS,
    lr: float = LEARNING_RATE,
    clip_norm: float = GRAD_CLIP_NORM,
    patience: int = PATIENCE,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> TrainingHistory:
    """Train a sentiment model with early stopping and gradient monitoring.

    Implements the full training loop including:
    - BCEWithLogitsLoss for binary classification
    - Adam optimizer with configurable learning rate
    - Gradient clipping to prevent exploding gradients
    - Validation evaluation per epoch
    - Early stopping when validation loss stops improving
    - Best checkpoint saving
    - Gradient L2 norm recording for the first 100 global steps

    Args:
        model: The SentimentRNN model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to train on (cuda, mps, or cpu).
        num_epochs: Maximum number of training epochs.
        lr: Learning rate for Adam optimizer.
        clip_norm: Maximum gradient norm for clipping.
        patience: Number of epochs without improvement before stopping.
        checkpoint_dir: Directory to save best model checkpoint.

    Returns:
        TrainingHistory with all recorded metrics.
    """
    history = TrainingHistory()

    # Move model to device
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping state
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint path based on model's rnn_type
    checkpoint_path = os.path.join(checkpoint_dir, f"{model.rnn_type}_best.pt")

    # Global step counter for gradient monitoring
    global_step = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Record gradient norms for the first GRADIENT_MONITOR_STEPS steps
            global_step += 1
            if global_step <= GRADIENT_MONITOR_STEPS:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        if name not in history.gradient_norms:
                            history.gradient_norms[name] = []
                        history.gradient_norms[name].append(grad_norm)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            # Optimizer step
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(logits) > 0.5).long()
            correct += (predictions == labels.long()).sum().item()
            total += labels.size(0)

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history.train_losses.append(epoch_train_loss)
        history.train_accs.append(epoch_train_acc)

        # --- Validation phase ---
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                logits = model(inputs).squeeze(1)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(logits) > 0.5).long()
                val_correct += (predictions == labels.long()).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        history.val_losses.append(epoch_val_loss)
        history.val_accs.append(epoch_val_acc)

        history.epochs_trained = epoch + 1

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        # --- Early stopping check ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            # Save best checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"  ✗ No improvement ({epochs_without_improvement}/{patience})"
            )
            if epochs_without_improvement >= patience:
                print(f"  Early stopping triggered after epoch {epoch + 1}")
                break

    history.training_time = time.time() - start_time

    print(
        f"\nTraining complete: {history.epochs_trained} epochs in "
        f"{history.training_time:.1f}s"
    )

    return history
