"""Tests for the evaluation module."""

import os
import tempfile
from dataclasses import fields

import numpy as np
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from torch.utils.data import DataLoader, TensorDataset

from evaluate import EvaluationResults, evaluate_model, print_comparison_table
from model import SentimentRNN


class TestEvaluationResults:
    """Tests for the EvaluationResults dataclass."""

    def test_creation_with_valid_values(self):
        """EvaluationResults can be created with valid metric values."""
        result = EvaluationResults(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1=0.85,
            confusion_matrix=[[10000, 2500], [1500, 11000]],
            rnn_type="LSTM",
        )
        assert result.accuracy == 0.85
        assert result.precision == 0.82
        assert result.recall == 0.88
        assert result.f1 == 0.85
        assert result.confusion_matrix == [[10000, 2500], [1500, 11000]]
        assert result.rnn_type == "LSTM"

    def test_creation_with_gru_type(self):
        """EvaluationResults works with GRU rnn_type."""
        result = EvaluationResults(
            accuracy=0.83,
            precision=0.80,
            recall=0.86,
            f1=0.83,
            confusion_matrix=[[9500, 3000], [2000, 10500]],
            rnn_type="GRU",
        )
        assert result.rnn_type == "GRU"

    def test_has_expected_fields(self):
        """EvaluationResults has exactly the expected fields."""
        field_names = {f.name for f in fields(EvaluationResults)}
        expected = {"accuracy", "precision", "recall", "f1", "confusion_matrix", "rnn_type"}
        assert field_names == expected

    def test_field_types(self):
        """EvaluationResults fields have correct type annotations."""
        field_types = {f.name: f.type for f in fields(EvaluationResults)}
        assert field_types["accuracy"] == float
        assert field_types["precision"] == float
        assert field_types["recall"] == float
        assert field_types["f1"] == float
        assert field_types["confusion_matrix"] == list[list[int]]
        assert field_types["rnn_type"] == str

    def test_is_dataclass(self):
        """EvaluationResults is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(EvaluationResults)

    def test_boundary_metric_values(self):
        """EvaluationResults accepts boundary values (0.0 and 1.0)."""
        result = EvaluationResults(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            confusion_matrix=[[0, 12500], [12500, 0]],
            rnn_type="LSTM",
        )
        assert result.accuracy == 0.0

        result_perfect = EvaluationResults(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            confusion_matrix=[[12500, 0], [0, 12500]],
            rnn_type="GRU",
        )
        assert result_perfect.accuracy == 1.0


def _create_model_and_checkpoint(rnn_type: str = "LSTM"):
    """Helper: create a small model, save checkpoint, return model + path + test data."""
    vocab_size = 100
    embedding_dim = 16
    hidden_size = 32
    num_layers = 2
    seq_len = 20
    num_samples = 64

    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        rnn_dropout=0.0,
        embed_dropout=0.0,
    )

    # Create a temporary checkpoint
    tmp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(tmp_dir, f"best_{rnn_type.lower()}.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # Create synthetic test data: half positive, half negative
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.cat([torch.zeros(num_samples // 2), torch.ones(num_samples // 2)]).long()
    dataset = TensorDataset(inputs, labels)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    return model, checkpoint_path, test_loader


class TestEvaluateModel:
    """Tests for the evaluate_model() function."""

    def test_returns_evaluation_results(self):
        """evaluate_model returns an EvaluationResults instance."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        assert isinstance(result, EvaluationResults)

    def test_metrics_in_valid_range(self):
        """All metrics are in [0, 1]."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix is 2x2 for binary classification."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2
        assert len(result.confusion_matrix[1]) == 2

    def test_confusion_matrix_sums_to_total_samples(self):
        """Confusion matrix entries sum to total number of samples."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        total = sum(sum(row) for row in result.confusion_matrix)
        assert total == 64  # num_samples in helper

    def test_rnn_type_is_set(self):
        """rnn_type field matches the provided argument."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("GRU")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "GRU")

        assert result.rnn_type == "GRU"

    def test_model_in_eval_mode_after_evaluation(self):
        """Model is in eval mode after evaluate_model completes."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        assert not model.training

    def test_deterministic_results(self):
        """Same checkpoint and data produce identical results."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result1 = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")
        result2 = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        assert result1.accuracy == result2.accuracy
        assert result1.precision == result2.precision
        assert result1.recall == result2.recall
        assert result1.f1 == result2.f1
        assert result1.confusion_matrix == result2.confusion_matrix


class TestPrintComparisonTable:
    """Tests for the print_comparison_table() function."""

    def _make_results(self, accuracy, precision, recall, f1, rnn_type):
        """Helper to create EvaluationResults with given metrics."""
        return EvaluationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=[[10000, 2500], [1500, 11000]],
            rnn_type=rnn_type,
        )

    def test_prints_header(self, capsys):
        """print_comparison_table outputs a header with 'LSTM vs GRU Comparison'."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.83, 0.80, 0.86, 0.83, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "LSTM vs GRU Comparison" in output

    def test_prints_all_metrics(self, capsys):
        """print_comparison_table displays Accuracy, Precision, Recall, and F1 Score."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.83, 0.80, 0.86, 0.83, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "Accuracy" in output
        assert "Precision" in output
        assert "Recall" in output
        assert "F1 Score" in output

    def test_prints_percentages(self, capsys):
        """Metrics are displayed as percentages (e.g., 85.00%)."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.83, 0.80, 0.86, 0.83, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "85.00%" in output
        assert "83.00%" in output

    def test_prints_separator_lines(self, capsys):
        """Output contains separator lines (=== and ---)."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.83, 0.80, 0.86, 0.83, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "=" * 60 in output
        assert "-" * 60 in output

    def test_prints_column_headers(self, capsys):
        """Output contains column headers: Metric, LSTM, GRU, Difference."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.83, 0.80, 0.86, 0.83, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "Metric" in output
        assert "LSTM" in output
        assert "GRU" in output
        assert "Difference" in output

    def test_positive_difference_has_plus_sign(self, capsys):
        """When LSTM > GRU, difference shows a '+' sign."""
        lstm = self._make_results(0.90, 0.90, 0.90, 0.90, "LSTM")
        gru = self._make_results(0.80, 0.80, 0.80, 0.80, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "+10.00%" in output

    def test_negative_difference_has_minus_sign(self, capsys):
        """When LSTM < GRU, difference shows a '-' sign."""
        lstm = self._make_results(0.75, 0.75, 0.75, 0.75, "LSTM")
        gru = self._make_results(0.85, 0.85, 0.85, 0.85, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "-10.00%" in output

    def test_zero_difference(self, capsys):
        """When both models have identical metrics, difference is +0.00%."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.85, 0.82, 0.88, 0.85, "GRU")

        print_comparison_table(lstm, gru)
        output = capsys.readouterr().out

        assert "+0.00%" in output

    def test_returns_none(self):
        """print_comparison_table returns None (prints to stdout only)."""
        lstm = self._make_results(0.85, 0.82, 0.88, 0.85, "LSTM")
        gru = self._make_results(0.83, 0.80, 0.86, 0.83, "GRU")

        result = print_comparison_table(lstm, gru)

        assert result is None


class TestMetricsIntegrity:
    """Tests for metrics integrity (Property 6).

    **Validates: Requirements 6.3, 6.4**
    """

    def test_accuracy_consistency_with_confusion_matrix(self):
        """Accuracy == (TP + TN) / (TP + TN + FP + FN) from confusion matrix."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        # confusion_matrix layout: [[TN, FP], [FN, TP]]
        tn, fp = result.confusion_matrix[0]
        fn, tp = result.confusion_matrix[1]
        total = tn + fp + fn + tp
        expected_accuracy = (tp + tn) / total

        assert abs(result.accuracy - expected_accuracy) < 1e-9

    def test_confusion_matrix_entries_non_negative_integers(self):
        """All entries in the confusion matrix are non-negative integers."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        for row in result.confusion_matrix:
            for entry in row:
                assert isinstance(entry, int), f"Entry {entry} is not an integer"
                assert entry >= 0, f"Entry {entry} is negative"

    def test_f1_is_harmonic_mean(self):
        """F1 ≈ 2 * (precision * recall) / (precision + recall) when precision + recall > 0."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        if result.precision + result.recall > 0:
            expected_f1 = (
                2 * (result.precision * result.recall) / (result.precision + result.recall)
            )
            assert abs(result.f1 - expected_f1) < 1e-9
        else:
            # When both precision and recall are 0, F1 should be 0
            assert result.f1 == 0.0

    def test_confusion_matrix_diagonal_dominance_when_accuracy_above_half(self):
        """For a model with accuracy > 0.5, diagonal sum > off-diagonal sum."""
        model, checkpoint_path, test_loader = _create_model_and_checkpoint("LSTM")
        device = torch.device("cpu")

        result = evaluate_model(model, test_loader, device, checkpoint_path, "LSTM")

        if result.accuracy > 0.5:
            tn, fp = result.confusion_matrix[0]
            fn, tp = result.confusion_matrix[1]
            diagonal_sum = tn + tp
            off_diagonal_sum = fp + fn
            assert diagonal_sum > off_diagonal_sum


class TestMetricsIntegrityProperty:
    """Property-based tests for metrics integrity (Property 6).

    **Validates: Requirements 6.3, 6.4**

    These tests use Hypothesis to verify metric relationships hold across
    many random binary classification scenarios.
    """

    @given(
        y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
        y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_metric_bounds_property(self, y_true, y_pred):
        """All metrics (accuracy, precision, recall, f1) are in [0, 1] for any binary predictions."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        assert 0.0 <= acc <= 1.0
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= f1 <= 1.0

    @given(
        y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
        y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_accuracy_consistency_property(self, y_true, y_pred):
        """Accuracy == (TP + TN) / total, computed from confusion matrix entries."""
        from sklearn.metrics import accuracy_score, confusion_matrix

        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # Need at least both classes present for a 2x2 confusion matrix
        assume(0 in y_true and 1 in y_true)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        tn, fp = cm[0]
        fn, tp = cm[1]
        total = tn + fp + fn + tp
        expected_acc = (tp + tn) / total

        assert abs(acc - expected_acc) < 1e-9

    @given(
        y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
        y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_confusion_matrix_non_negative_and_sums_to_total(self, y_true, y_pred):
        """Confusion matrix entries are non-negative and sum to total samples."""
        from sklearn.metrics import confusion_matrix

        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        assume(0 in y_true and 1 in y_true)

        cm = confusion_matrix(y_true, y_pred)

        # All entries non-negative
        assert (cm >= 0).all()
        # Sum equals total samples
        assert cm.sum() == min_len

    @given(
        y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
        y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_f1_harmonic_mean_property(self, y_true, y_pred):
        """F1 ≈ 2 * (precision * recall) / (precision + recall) when precision + recall > 0."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if prec + rec > 0:
            expected_f1 = 2 * (prec * rec) / (prec + rec)
            assert abs(f1 - expected_f1) < 1e-9
        else:
            assert f1 == 0.0

    @given(
        y_true=st.lists(st.integers(min_value=0, max_value=1), min_size=20, max_size=200),
        y_pred=st.lists(st.integers(min_value=0, max_value=1), min_size=20, max_size=200),
    )
    @settings(max_examples=100, deadline=None)
    def test_diagonal_dominance_when_accuracy_above_half(self, y_true, y_pred):
        """For accuracy > 0.5, confusion matrix diagonal sum > off-diagonal sum."""
        from sklearn.metrics import accuracy_score, confusion_matrix

        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        assume(0 in y_true and 1 in y_true)

        acc = accuracy_score(y_true, y_pred)
        assume(acc > 0.5)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0]
        fn, tp = cm[1]

        diagonal_sum = tn + tp
        off_diagonal_sum = fp + fn
        assert diagonal_sum > off_diagonal_sum
