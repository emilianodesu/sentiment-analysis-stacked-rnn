"""Stacked RNN model definition (LSTM/GRU)."""

import torch.nn as nn
from torch import Tensor


class SentimentRNN(nn.Module):
    """Configurable Stacked RNN for binary sentiment classification.

    Supports both LSTM and GRU recurrent layer types with identical interface.

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens).
        embedding_dim: Dimension of the embedding vectors.
        hidden_size: Number of features in the hidden state of the RNN.
        num_layers: Number of stacked recurrent layers.
        rnn_type: Type of recurrent layer ("LSTM" or "GRU").
        rnn_dropout: Dropout between stacked RNN layers.
        embed_dropout: Dropout applied after the embedding layer.

    Raises:
        ValueError: If rnn_type is not "LSTM" or "GRU".
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        rnn_type: str,
        rnn_dropout: float,
        embed_dropout: float,
    ) -> None:
        super().__init__()

        if rnn_type not in ("LSTM", "GRU"):
            raise ValueError(
                f"Invalid rnn_type '{rnn_type}'. Supported options: 'LSTM', 'GRU'"
            )

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Dropout after embedding
        self.embed_dropout = nn.Dropout(p=embed_dropout)

        # Stacked RNN (LSTM or GRU)
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        # Initialize LSTM forget gate bias to 1.0 to encourage remembering
        # early in training (Jozefowicz et al., 2015)
        if rnn_type == "LSTM":
            for name, param in self.rnn.named_parameters():
                if "bias" in name:
                    # LSTM bias is [input_gate, forget_gate, cell_gate, output_gate]
                    # each of size hidden_size. Set forget gate bias to 1.0.
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

        # Output linear layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of token indices with shape (batch_size, seq_len).

        Returns:
            Logits tensor with shape (batch_size, 1).
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Apply dropout after embedding
        embedded = self.embed_dropout(embedded)

        # RNN forward pass
        # output: (batch_size, seq_len, hidden_size)
        # For LSTM: hidden = (h_n, c_n) where h_n shape is (num_layers, batch_size, hidden_size)
        # For GRU: hidden = h_n with shape (num_layers, batch_size, hidden_size)
        _output, hidden = self.rnn(embedded)

        # Extract hidden state from the final layer at the last time step
        if self.rnn_type == "LSTM":
            # h_n shape: (num_layers, batch_size, hidden_size)
            final_hidden = hidden[0][-1]  # (batch_size, hidden_size)
        else:
            # h_n shape: (num_layers, batch_size, hidden_size)
            final_hidden = hidden[-1]  # (batch_size, hidden_size)

        # Linear layer: (batch_size, hidden_size) -> (batch_size, 1)
        logits = self.fc(final_hidden)

        return logits


def print_model_summary(model: SentimentRNN) -> None:
    """Print model architecture AND trainable parameters per layer together.

    Displays the full architecture followed by a breakdown of trainable
    parameters for each named layer. Both sections are always printed
    together and SHALL NOT be shown independently.

    Args:
        model: A SentimentRNN model instance.
    """
    separator = "=" * 60

    print(separator)
    print("MODEL SUMMARY")
    print(separator)

    # Architecture section
    print("\n--- Architecture ---\n")
    print(model)

    # Parameters section
    print("\n--- Trainable Parameters ---\n")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"  {name:<40} {num_params:>12,}")

    print(f"\n  {'TOTAL':<40} {total_params:>12,}")
    print(separator)
