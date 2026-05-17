"""Tests for the main pipeline orchestrator."""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from data import IMDBDataset
from evaluate import evaluate_model
from main import verify_setup
from model import SentimentRNN
from train import set_seed, train_model, get_device


class TestVerifySetup:
    """Tests for verify_setup function."""

    def test_passes_in_project_root(self):
        """Test that verify_setup passes when run from the project root."""
        # This test runs from the project root where all modules exist
        verify_setup()

    def test_raises_on_missing_module(self, tmp_path, monkeypatch):
        """Test that verify_setup raises RuntimeError for missing modules."""
        import main as main_module

        monkeypatch.setattr(main_module, "__file__", str(tmp_path / "main.py"))
        with pytest.raises(RuntimeError, match="Missing required module files"):
            verify_setup()


class TestIntegrationPipeline:
    """Integration test with small subset (100 samples, 2 epochs)."""

    @pytest.fixture
    def small_data(self):
        """Create a small synthetic dataset for integration testing."""
        set_seed(config.SEED)
        num_samples = 100
        seq_len = config.MAX_LEN

        # Generate random sequences and binary labels
        sequences = torch.randint(
            0, config.VOCAB_SIZE, (num_samples, seq_len), dtype=torch.long
        )
        labels = torch.randint(0, 2, (num_samples,)).float()

        dataset = TensorDataset(sequences, labels)

        # Split: 60 train, 20 val, 20 test
        train_dataset = TensorDataset(sequences[:60], labels[:60])
        val_dataset = TensorDataset(sequences[60:80], labels[60:80])
        test_dataset = TensorDataset(sequences[80:], labels[80:])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        return train_loader, val_loader, test_loader

    def test_full_pipeline_lstm(self, small_data, tmp_path):
        """Test full pipeline with LSTM on small data (2 epochs)."""
        train_loader, val_loader, test_loader = small_data
        device = torch.device("cpu")

        set_seed(config.SEED)

        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        checkpoint_dir = str(tmp_path / "checkpoints")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=2,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify training history
        assert history.epochs_trained == 2
        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 2
        assert len(history.train_accs) == 2
        assert len(history.val_accs) == 2
        assert history.training_time > 0

        # Verify checkpoint was saved
        checkpoint_path = os.path.join(checkpoint_dir, "LSTM_best.pt")
        assert os.path.exists(checkpoint_path)

        # Verify evaluation works
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            checkpoint_path=checkpoint_path,
            rnn_type="LSTM",
        )

        assert 0.0 <= results.accuracy <= 1.0
        assert 0.0 <= results.f1 <= 1.0
        assert results.rnn_type == "LSTM"

    def test_full_pipeline_gru(self, small_data, tmp_path):
        """Test full pipeline with GRU on small data (2 epochs)."""
        train_loader, val_loader, test_loader = small_data
        device = torch.device("cpu")

        set_seed(config.SEED)

        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        checkpoint_dir = str(tmp_path / "checkpoints")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=2,
            checkpoint_dir=checkpoint_dir,
        )

        # Verify training history
        assert history.epochs_trained == 2
        assert len(history.train_losses) == 2
        assert history.training_time > 0

        # Verify checkpoint was saved
        checkpoint_path = os.path.join(checkpoint_dir, "GRU_best.pt")
        assert os.path.exists(checkpoint_path)

        # Verify evaluation works
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            checkpoint_path=checkpoint_path,
            rnn_type="GRU",
        )

        assert 0.0 <= results.accuracy <= 1.0
        assert results.rnn_type == "GRU"

    def test_gradient_norms_recorded(self, small_data, tmp_path):
        """Test that gradient norms are recorded during training."""
        train_loader, val_loader, _ = small_data
        device = torch.device("cpu")

        set_seed(config.SEED)

        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        checkpoint_dir = str(tmp_path / "checkpoints")

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=2,
            checkpoint_dir=checkpoint_dir,
        )

        # Gradient norms should be recorded for named parameters
        assert len(history.gradient_norms) > 0
        for layer_name, norms in history.gradient_norms.items():
            assert len(norms) > 0
            assert all(n >= 0 for n in norms)
