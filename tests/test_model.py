"""Tests for model.py — SentimentRNN and print_model_summary.

Validates Property 4 (Model Consistency) and error handling (REQ-4.6).
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from model import SentimentRNN, print_model_summary
import config


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestSentimentRNNConstruction:
    """Test model instantiation and configuration."""

    def test_lstm_model_creates_successfully(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        assert model.rnn_type == "LSTM"
        assert isinstance(model.rnn, torch.nn.LSTM)

    def test_gru_model_creates_successfully(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        assert model.rnn_type == "GRU"
        assert isinstance(model.rnn, torch.nn.GRU)

    def test_invalid_rnn_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Supported options"):
            SentimentRNN(
                vocab_size=config.VOCAB_SIZE,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                rnn_type="RNN",
                rnn_dropout=config.RNN_DROPOUT,
                embed_dropout=config.EMBED_DROPOUT,
            )

    def test_invalid_rnn_type_empty_string(self):
        with pytest.raises(ValueError):
            SentimentRNN(
                vocab_size=config.VOCAB_SIZE,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                rnn_type="",
                rnn_dropout=config.RNN_DROPOUT,
                embed_dropout=config.EMBED_DROPOUT,
            )

    def test_invalid_rnn_type_lowercase(self):
        """rnn_type must be exact case: 'LSTM' or 'GRU'."""
        with pytest.raises(ValueError):
            SentimentRNN(
                vocab_size=config.VOCAB_SIZE,
                embedding_dim=config.EMBEDDING_DIM,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                rnn_type="lstm",
                rnn_dropout=config.RNN_DROPOUT,
                embed_dropout=config.EMBED_DROPOUT,
            )

    def test_embedding_dimensions(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        assert model.embedding.num_embeddings == config.VOCAB_SIZE
        assert model.embedding.embedding_dim == config.EMBEDDING_DIM

    def test_linear_layer_dimensions(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        assert model.fc.in_features == config.HIDDEN_SIZE
        assert model.fc.out_features == 1


class TestSentimentRNNForward:
    """Test forward pass behavior."""

    def test_lstm_output_shape(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        model.eval()
        x = torch.randint(0, config.VOCAB_SIZE, (config.BATCH_SIZE, config.MAX_LEN))
        with torch.no_grad():
            output = model(x)
        assert output.shape == (config.BATCH_SIZE, 1)

    def test_gru_output_shape(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        model.eval()
        x = torch.randint(0, config.VOCAB_SIZE, (config.BATCH_SIZE, config.MAX_LEN))
        with torch.no_grad():
            output = model(x)
        assert output.shape == (config.BATCH_SIZE, 1)

    def test_single_sample_output_shape(self):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        model.eval()
        x = torch.randint(0, config.VOCAB_SIZE, (1, config.MAX_LEN))
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 1)


class TestModelParameters:
    """Test parameter counts and comparisons."""

    def test_gru_has_fewer_params_than_lstm(self):
        """GRU has 3 gates vs LSTM's 4 gates, so fewer parameters."""
        lstm = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        gru = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        lstm_params = sum(p.numel() for p in lstm.parameters())
        gru_params = sum(p.numel() for p in gru.parameters())
        assert gru_params < lstm_params


class TestPrintModelSummary:
    """Test print_model_summary function."""

    def test_prints_architecture_and_params(self, capsys):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        print_model_summary(model)
        captured = capsys.readouterr()

        # Must contain both architecture and parameters sections
        assert "Architecture" in captured.out
        assert "Trainable Parameters" in captured.out
        assert "TOTAL" in captured.out
        assert "SentimentRNN" in captured.out
        assert "Embedding" in captured.out

    def test_prints_for_gru(self, capsys):
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        print_model_summary(model)
        captured = capsys.readouterr()

        assert "Architecture" in captured.out
        assert "Trainable Parameters" in captured.out
        assert "GRU" in captured.out


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis) — Property 4: Model Consistency
# ---------------------------------------------------------------------------


class TestProperty4ModelConsistency:
    """Property 4: Model Consistency.

    **Validates: Requirements 4.1, 4.2, 4.4**

    - For rnn_type="LSTM" and rnn_type="GRU": output shape is always (batch_size, 1)
    - GRU has fewer parameters than LSTM
    - Both models accept the same input and produce output of the same shape
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_lstm_output_shape_property(self, batch_size: int, seq_len: int):
        """LSTM output shape is always (batch_size, 1) for any valid input.

        **Validates: Requirements 4.1, 4.4**
        """
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        model.eval()
        x = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len))
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 1)

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_gru_output_shape_property(self, batch_size: int, seq_len: int):
        """GRU output shape is always (batch_size, 1) for any valid input.

        **Validates: Requirements 4.1, 4.4**
        """
        model = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        model.eval()
        x = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len))
        with torch.no_grad():
            output = model(x)
        assert output.shape == (batch_size, 1)

    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        seq_len=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_both_models_same_output_shape(self, batch_size: int, seq_len: int):
        """Both LSTM and GRU produce the same output shape for identical input.

        **Validates: Requirements 4.1, 4.2, 4.4**
        """
        lstm = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="LSTM",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        gru = SentimentRNN(
            vocab_size=config.VOCAB_SIZE,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            rnn_type="GRU",
            rnn_dropout=config.RNN_DROPOUT,
            embed_dropout=config.EMBED_DROPOUT,
        )
        lstm.eval()
        gru.eval()
        x = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len))
        with torch.no_grad():
            lstm_out = lstm(x)
            gru_out = gru(x)
        assert lstm_out.shape == gru_out.shape == (batch_size, 1)
