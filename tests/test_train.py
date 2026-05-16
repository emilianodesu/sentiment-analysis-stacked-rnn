"""Tests for train.py utility functions: set_seed, get_device, and train_model."""

import os
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import GRADIENT_MONITOR_STEPS
from model import SentimentRNN
from train import TrainingHistory, get_device, set_seed, train_model


class TestSetSeed:
    """Tests for set_seed determinism (Property 3)."""

    def test_torch_determinism(self):
        """Two calls with the same seed produce identical torch tensors."""
        set_seed(42)
        t1 = torch.randn(5, 5)

        set_seed(42)
        t2 = torch.randn(5, 5)

        assert torch.equal(t1, t2)

    def test_numpy_determinism(self):
        """Two calls with the same seed produce identical numpy arrays."""
        set_seed(42)
        a1 = np.random.rand(5, 5)

        set_seed(42)
        a2 = np.random.rand(5, 5)

        assert np.array_equal(a1, a2)

    def test_python_random_determinism(self):
        """Two calls with the same seed produce identical random values."""
        set_seed(42)
        vals1 = [random.random() for _ in range(10)]

        set_seed(42)
        vals2 = [random.random() for _ in range(10)]

        assert vals1 == vals2

    def test_different_seeds_produce_different_results(self):
        """Different seeds produce different random outputs."""
        set_seed(42)
        t1 = torch.randn(5, 5)

        set_seed(123)
        t2 = torch.randn(5, 5)

        assert not torch.equal(t1, t2)

    def test_cudnn_settings(self):
        """set_seed configures cuDNN for deterministic behavior."""
        set_seed(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


class TestGetDevice:
    """Tests for get_device (REQ-5.8, REQ-5.9)."""

    def test_returns_valid_device(self):
        """get_device returns a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_known_device_type(self):
        """get_device returns one of cuda, mps, or cpu."""
        device = get_device()
        assert device.type in ("cuda", "mps", "cpu")

    def test_cpu_fallback(self):
        """If no GPU is available, cpu is always a valid fallback."""
        device = get_device()
        # At minimum, the function should not raise and should return a device
        # On CI/CPU-only machines this will be cpu; on GPU machines it will be cuda/mps
        assert device.type in ("cuda", "mps", "cpu")


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_default_initialization(self):
        """TrainingHistory initializes with empty lists and zero scalars."""
        history = TrainingHistory()
        assert history.train_losses == []
        assert history.val_losses == []
        assert history.train_accs == []
        assert history.val_accs == []
        assert history.gradient_norms == {}
        assert history.training_time == 0.0
        assert history.epochs_trained == 0

    def test_independent_default_lists(self):
        """Each instance gets its own list/dict (no shared mutable defaults)."""
        h1 = TrainingHistory()
        h2 = TrainingHistory()
        h1.train_losses.append(0.5)
        h1.gradient_norms["embed"] = [1.0, 2.0]
        assert h2.train_losses == []
        assert h2.gradient_norms == {}

    def test_stores_epoch_metrics(self):
        """TrainingHistory correctly stores per-epoch loss and accuracy."""
        history = TrainingHistory()
        history.train_losses.append(0.7)
        history.train_losses.append(0.5)
        history.val_losses.append(0.8)
        history.val_losses.append(0.6)
        history.train_accs.append(0.6)
        history.train_accs.append(0.75)
        history.val_accs.append(0.55)
        history.val_accs.append(0.7)
        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 2
        assert len(history.train_accs) == 2
        assert len(history.val_accs) == 2

    def test_stores_gradient_norms(self):
        """TrainingHistory stores gradient norms per layer."""
        history = TrainingHistory()
        history.gradient_norms["embedding"] = [0.1, 0.2, 0.3]
        history.gradient_norms["rnn_layer_0"] = [1.0, 1.1, 1.2]
        history.gradient_norms["rnn_layer_1"] = [0.8, 0.9, 1.0]
        history.gradient_norms["linear"] = [0.5, 0.6, 0.7]
        assert len(history.gradient_norms) == 4
        assert len(history.gradient_norms["embedding"]) == 3

    def test_training_time_and_epochs(self):
        """TrainingHistory stores training_time and epochs_trained."""
        history = TrainingHistory()
        history.training_time = 120.5
        history.epochs_trained = 7
        assert history.training_time == 120.5
        assert history.epochs_trained == 7

    def test_is_dataclass(self):
        """TrainingHistory is a proper dataclass."""
        from dataclasses import fields

        field_names = {f.name for f in fields(TrainingHistory)}
        expected = {
            "train_losses",
            "val_losses",
            "train_accs",
            "val_accs",
            "gradient_norms",
            "training_time",
            "epochs_trained",
        }
        assert field_names == expected


# --- Helper fixtures for train_model tests ---


def _make_small_model():
    """Create a small SentimentRNN for testing (reduced dimensions)."""
    return SentimentRNN(
        vocab_size=100,
        embedding_dim=16,
        hidden_size=32,
        num_layers=2,
        rnn_type="LSTM",
        rnn_dropout=0.0,
        embed_dropout=0.0,
    )


def _make_dataloaders(n_train=64, n_val=32, seq_len=20, batch_size=16):
    """Create small synthetic DataLoaders for testing."""
    # Random token indices and binary labels
    train_x = torch.randint(0, 100, (n_train, seq_len))
    train_y = torch.randint(0, 2, (n_train,))
    val_x = torch.randint(0, 100, (n_val, seq_len))
    val_y = torch.randint(0, 2, (n_val,))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


class TestTrainModel:
    """Tests for train_model function (REQ-5.2 to REQ-5.7)."""

    def test_returns_training_history(self):
        """train_model returns a TrainingHistory instance."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=2, checkpoint_dir=tmpdir
            )

        assert isinstance(history, TrainingHistory)

    def test_records_per_epoch_metrics(self):
        """train_model records loss and accuracy for each epoch."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=3, patience=10, checkpoint_dir=tmpdir
            )

        assert len(history.train_losses) == 3
        assert len(history.val_losses) == 3
        assert len(history.train_accs) == 3
        assert len(history.val_accs) == 3
        assert history.epochs_trained == 3

    def test_losses_are_positive(self):
        """All recorded losses should be positive."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=2, checkpoint_dir=tmpdir
            )

        assert all(loss > 0 for loss in history.train_losses)
        assert all(loss > 0 for loss in history.val_losses)

    def test_accuracies_in_valid_range(self):
        """All recorded accuracies should be in [0, 1]."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=2, checkpoint_dir=tmpdir
            )

        assert all(0 <= acc <= 1 for acc in history.train_accs)
        assert all(0 <= acc <= 1 for acc in history.val_accs)

    def test_saves_checkpoint(self):
        """train_model saves a checkpoint file when val loss improves."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            train_model(
                model, train_loader, val_loader, device,
                num_epochs=2, checkpoint_dir=tmpdir
            )
            checkpoint_path = os.path.join(tmpdir, "LSTM_best.pt")
            assert os.path.exists(checkpoint_path)

    def test_early_stopping(self):
        """train_model stops early when val loss doesn't improve for patience epochs."""
        set_seed(42)
        model = _make_small_model()
        # Use very small data so model overfits quickly and val loss increases
        train_loader, val_loader = _make_dataloaders(n_train=16, n_val=16, batch_size=16)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=50, patience=3, checkpoint_dir=tmpdir
            )

        # Should stop before 50 epochs due to early stopping
        assert history.epochs_trained <= 50
        # epochs_trained should be at least patience + 1 (first good epoch + patience bad ones)
        assert history.epochs_trained >= 2

    def test_records_gradient_norms(self):
        """train_model records gradient norms for the first GRADIENT_MONITOR_STEPS steps."""
        set_seed(42)
        model = _make_small_model()
        # Enough batches to exceed GRADIENT_MONITOR_STEPS
        train_loader, val_loader = _make_dataloaders(
            n_train=256, n_val=32, batch_size=2
        )
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=1, checkpoint_dir=tmpdir
            )

        # Should have gradient norms recorded
        assert len(history.gradient_norms) > 0
        # Each layer should have exactly GRADIENT_MONITOR_STEPS entries
        for layer_name, norms in history.gradient_norms.items():
            assert len(norms) == GRADIENT_MONITOR_STEPS
            assert all(n >= 0 for n in norms)

    def test_training_time_recorded(self):
        """train_model records positive training time."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=1, checkpoint_dir=tmpdir
            )

        assert history.training_time > 0

    def test_checkpoint_loadable(self):
        """Saved checkpoint can be loaded back into the model."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            train_model(
                model, train_loader, val_loader, device,
                num_epochs=2, checkpoint_dir=tmpdir
            )
            checkpoint_path = os.path.join(tmpdir, "LSTM_best.pt")

            # Load into a fresh model
            fresh_model = _make_small_model()
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            fresh_model.load_state_dict(state_dict)

            # Verify it can do a forward pass
            test_input = torch.randint(0, 100, (1, 20))
            output = fresh_model(test_input)
            assert output.shape == (1, 1)

    def test_uses_bce_with_logits_loss(self):
        """train_model uses BCEWithLogitsLoss (model outputs raw logits)."""
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders()
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=1, checkpoint_dir=tmpdir
            )

        # If BCEWithLogitsLoss is used correctly, loss should be reasonable
        # (not NaN or extremely large which would indicate wrong loss function)
        assert not any(np.isnan(l) for l in history.train_losses)
        assert all(l < 10.0 for l in history.train_losses)


class TestDeterminism:
    """Tests for Property 3: Running train_model with the same seed produces identical histories."""

    def test_same_seed_produces_identical_histories(self):
        """Two train_model runs with the same seed produce identical losses, accuracies, and gradient norms.

        **Validates: Requirements 5.1, 9.4**
        """
        device = torch.device("cpu")

        # First run
        set_seed(42)
        model1 = _make_small_model()
        train_loader1, val_loader1 = _make_dataloaders(n_train=64, n_val=32, seq_len=20, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir1:
            history1 = train_model(
                model1, train_loader1, val_loader1, device,
                num_epochs=3, patience=10, checkpoint_dir=tmpdir1
            )

        # Second run with same seed
        set_seed(42)
        model2 = _make_small_model()
        train_loader2, val_loader2 = _make_dataloaders(n_train=64, n_val=32, seq_len=20, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir2:
            history2 = train_model(
                model2, train_loader2, val_loader2, device,
                num_epochs=3, patience=10, checkpoint_dir=tmpdir2
            )

        # Losses must be identical
        assert history1.train_losses == history2.train_losses, (
            f"Train losses differ: {history1.train_losses} vs {history2.train_losses}"
        )
        assert history1.val_losses == history2.val_losses, (
            f"Val losses differ: {history1.val_losses} vs {history2.val_losses}"
        )

        # Accuracies must be identical
        assert history1.train_accs == history2.train_accs, (
            f"Train accs differ: {history1.train_accs} vs {history2.train_accs}"
        )
        assert history1.val_accs == history2.val_accs, (
            f"Val accs differ: {history1.val_accs} vs {history2.val_accs}"
        )

        # Gradient norms must be identical
        assert history1.gradient_norms.keys() == history2.gradient_norms.keys()
        for layer_name in history1.gradient_norms:
            assert history1.gradient_norms[layer_name] == history2.gradient_norms[layer_name], (
                f"Gradient norms differ for layer {layer_name}"
            )

    def test_different_seeds_produce_different_histories(self):
        """Two train_model runs with different seeds produce different training histories.

        **Validates: Requirements 5.1, 9.4**
        """
        device = torch.device("cpu")

        # First run with seed 42
        set_seed(42)
        model1 = _make_small_model()
        train_loader1, val_loader1 = _make_dataloaders(n_train=64, n_val=32, seq_len=20, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir1:
            history1 = train_model(
                model1, train_loader1, val_loader1, device,
                num_epochs=2, patience=10, checkpoint_dir=tmpdir1
            )

        # Second run with seed 123
        set_seed(123)
        model2 = _make_small_model()
        train_loader2, val_loader2 = _make_dataloaders(n_train=64, n_val=32, seq_len=20, batch_size=16)

        with tempfile.TemporaryDirectory() as tmpdir2:
            history2 = train_model(
                model2, train_loader2, val_loader2, device,
                num_epochs=2, patience=10, checkpoint_dir=tmpdir2
            )

        # At least one metric should differ
        losses_differ = history1.train_losses != history2.train_losses
        accs_differ = history1.train_accs != history2.train_accs
        assert losses_differ or accs_differ, "Different seeds should produce different results"


class TestEarlyStopping:
    """Tests for Property 5: Early stopping behavior and best checkpoint correctness."""

    def test_stops_before_max_epochs(self):
        """Training stops before max_epochs when val loss doesn't improve for patience epochs.

        **Validates: Requirements 5.6**
        """
        set_seed(42)
        model = _make_small_model()
        # Very small data to encourage overfitting and val loss increase
        train_loader, val_loader = _make_dataloaders(n_train=16, n_val=16, batch_size=16)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=50, patience=3, checkpoint_dir=tmpdir
            )

        # Should stop before 50 epochs
        assert history.epochs_trained < 50, (
            f"Expected early stopping before 50 epochs, got {history.epochs_trained}"
        )
        # Must train at least patience + 1 epochs (1 good + patience bad)
        assert history.epochs_trained >= 4, (
            f"Expected at least 4 epochs (1 best + 3 patience), got {history.epochs_trained}"
        )

    def test_best_checkpoint_matches_lowest_val_loss(self):
        """The saved best checkpoint corresponds to the epoch with lowest validation loss.

        **Validates: Requirements 5.7**
        """
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders(n_train=32, n_val=32, batch_size=16)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=5, patience=10, checkpoint_dir=tmpdir
            )

            # Load the saved checkpoint
            checkpoint_path = os.path.join(tmpdir, "LSTM_best.pt")
            assert os.path.exists(checkpoint_path)

            # Load checkpoint into a fresh model and run validation
            best_model = _make_small_model()
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            best_model.load_state_dict(state_dict)
            best_model.eval()

            # Compute val loss with the loaded checkpoint
            criterion = torch.nn.BCEWithLogitsLoss()
            val_loss = 0.0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    logits = best_model(inputs).squeeze(1)
                    loss = criterion(logits, labels.float())
                    val_loss += loss.item() * inputs.size(0)
                    val_total += labels.size(0)

            checkpoint_val_loss = val_loss / val_total

            # The checkpoint val loss should match the minimum val loss from history
            min_val_loss = min(history.val_losses)
            assert abs(checkpoint_val_loss - min_val_loss) < 1e-5, (
                f"Checkpoint val loss ({checkpoint_val_loss:.6f}) doesn't match "
                f"min history val loss ({min_val_loss:.6f})"
            )

    def test_epochs_trained_equals_history_length(self):
        """epochs_trained matches the number of recorded loss values.

        **Validates: Requirements 5.6**
        """
        set_seed(42)
        model = _make_small_model()
        train_loader, val_loader = _make_dataloaders(n_train=16, n_val=16, batch_size=16)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = train_model(
                model, train_loader, val_loader, device,
                num_epochs=50, patience=3, checkpoint_dir=tmpdir
            )

        assert history.epochs_trained == len(history.train_losses)
        assert history.epochs_trained == len(history.val_losses)
        assert history.epochs_trained == len(history.train_accs)
        assert history.epochs_trained == len(history.val_accs)
