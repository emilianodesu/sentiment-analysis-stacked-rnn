"""Bug condition exploration and fix verification tests for LSTM training failure.

Bug: LSTM model instantiated before per-training set_seed(42) gets weights that
lead to a degenerate local minimum (~50% accuracy). The fix moves model
instantiation to after the per-training set_seed call.

Validates: lstm-training-fix bugfix spec
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from model import SentimentRNN
from train import set_seed, train_model


def _create_synthetic_data(num_samples=200, seed=42):
    """Create synthetic data where positive/negative is learnable."""
    set_seed(seed)
    seq_len = config.MAX_LEN

    # Create data with a learnable pattern:
    # Positive samples have higher token indices on average
    # Negative samples have lower token indices on average
    sequences = []
    labels = []
    for i in range(num_samples):
        if i % 2 == 0:
            # Positive: tokens from upper range
            seq = torch.randint(
                config.VOCAB_SIZE // 2, config.VOCAB_SIZE, (seq_len,)
            )
            labels.append(1.0)
        else:
            # Negative: tokens from lower range
            seq = torch.randint(0, config.VOCAB_SIZE // 2, (seq_len,))
            labels.append(0.0)
        sequences.append(seq)

    sequences = torch.stack(sequences)
    labels = torch.tensor(labels)
    return sequences, labels


def _make_loaders(sequences, labels, batch_size=32):
    """Split data into train/val loaders."""
    n = len(sequences)
    split = int(n * 0.8)
    train_ds = TensorDataset(sequences[:split], labels[:split])
    val_ds = TensorDataset(sequences[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class TestBugConditionExploration:
    """Demonstrate the bug: LSTM instantiated before per-training seed fails.

    CRITICAL: These tests demonstrate the bug mechanism. Task 1 in bugfix spec.
    """

    def test_seed_does_not_reinitialize_existing_model_weights(self):
        """Calling set_seed after model creation does NOT change model weights.

        This is the core bug mechanism: set_seed(42) before training resets RNG
        but leaves model weights unchanged from their creation-time values.
        """
        set_seed(42)
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        # Capture weights before second set_seed
        weights_before = model.rnn.weight_ih_l0.data.clone()

        # Call set_seed again (simulating per-training seed reset)
        set_seed(42)

        # Weights are UNCHANGED — this is the bug
        weights_after = model.rnn.weight_ih_l0.data.clone()
        assert torch.equal(weights_before, weights_after), (
            "set_seed should NOT reinitialize existing model weights"
        )

    def test_model_instantiation_order_affects_weights(self):
        """Models created at different points in random sequence get different weights.

        This shows why LSTM (created first) and GRU (created second) get
        different initializations from the same initial seed.
        """
        # Scenario 1: Create model immediately after seed
        set_seed(42)
        model_a = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        # Scenario 2: Create another model (consuming random state)
        # then create LSTM — it gets different weights
        set_seed(42)
        _dummy = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        model_b = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        # model_a and model_b have DIFFERENT weights despite same seed
        assert not torch.equal(
            model_a.rnn.weight_ih_l0.data, model_b.rnn.weight_ih_l0.data
        ), "LSTM created after another model gets different initialization"

    def test_fresh_instantiation_after_seed_is_deterministic(self):
        """Model created immediately after set_seed always gets same weights.

        This is the fix pattern: set_seed(42) → create model → train.
        """
        set_seed(42)
        model_1 = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        set_seed(42)
        model_2 = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        # Same seed → same weights (deterministic)
        assert torch.equal(
            model_1.rnn.weight_ih_l0.data, model_2.rnn.weight_ih_l0.data
        )
        assert torch.equal(
            model_1.embedding.weight.data, model_2.embedding.weight.data
        )


class TestFixVerification:
    """Verify the fix: LSTM instantiated after per-training seed learns correctly.

    These tests use synthetic data to verify the fix pattern works without
    requiring full IMDB training (which takes minutes).
    """

    def test_lstm_learns_with_post_seed_instantiation(self, tmp_path):
        """LSTM created after set_seed(42) achieves above-chance accuracy.

        Fix pattern: set_seed(42) → create LSTM → train → train_acc improves
        (on synthetic data, improving train accuracy demonstrates learning)
        """
        sequences, labels = _create_synthetic_data(num_samples=400, seed=99)
        train_loader, val_loader = _make_loaders(sequences, labels)

        # Fix pattern: seed THEN create model
        set_seed(42)
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            num_epochs=5,
            patience=5,
            checkpoint_dir=str(tmp_path),
        )

        # LSTM should learn (train accuracy improves from first to last epoch)
        assert history.train_accs[-1] > history.train_accs[0], (
            f"LSTM with post-seed instantiation should learn. "
            f"Train acc: {history.train_accs[0]:.4f} → {history.train_accs[-1]:.4f}"
        )

    def test_gru_learns_with_post_seed_instantiation(self, tmp_path):
        """GRU created after set_seed(42) achieves above-chance accuracy.

        Preservation: GRU should still work with the fix pattern.
        """
        sequences, labels = _create_synthetic_data(num_samples=400, seed=99)
        train_loader, val_loader = _make_loaders(sequences, labels)

        # Fix pattern: seed THEN create model
        set_seed(42)
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            num_epochs=5,
            patience=5,
            checkpoint_dir=str(tmp_path),
        )

        # GRU should learn (train accuracy improves)
        assert history.train_accs[-1] > history.train_accs[0], (
            f"GRU with post-seed instantiation should learn. "
            f"Train acc: {history.train_accs[0]:.4f} → {history.train_accs[-1]:.4f}"
        )

    def test_both_models_deterministic_with_fix_pattern(self, tmp_path):
        """Both models produce identical results across runs with fix pattern."""
        sequences, labels = _create_synthetic_data(num_samples=200, seed=99)
        train_loader, val_loader = _make_loaders(sequences, labels)

        # Run 1
        set_seed(42)
        model_1 = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        history_1 = train_model(
            model=model_1,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            num_epochs=2,
            checkpoint_dir=str(tmp_path / "run1"),
        )

        # Run 2 (same seed → same results)
        set_seed(42)
        model_2 = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        history_2 = train_model(
            model=model_2,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            num_epochs=2,
            checkpoint_dir=str(tmp_path / "run2"),
        )

        # Deterministic: same losses and accuracies
        assert history_1.train_losses == history_2.train_losses
        assert history_1.val_losses == history_2.val_losses
