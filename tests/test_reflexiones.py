"""Tests for the reflexiones module."""
# pylint: disable=redefined-outer-name

import os
import tempfile

import pytest

from evaluate import EvaluationResults
from reflexiones import (
    _compute_avg_gradient_norm,
    _compute_layer_gradient_stats,
    _detect_gradient_issues,
    generate_reflexiones,
)
from train import TrainingHistory


@pytest.fixture
def lstm_history():
    """Create a sample LSTM training history."""
    return TrainingHistory(
        train_losses=[0.65, 0.45, 0.35, 0.28, 0.22],
        val_losses=[0.60, 0.42, 0.38, 0.39, 0.40],
        train_accs=[0.62, 0.78, 0.84, 0.88, 0.91],
        val_accs=[0.65, 0.80, 0.83, 0.82, 0.81],
        gradient_norms={
            "embedding.weight": [0.5, 0.4, 0.3, 0.35, 0.32],
            "rnn.weight_ih_l0": [1.2, 1.0, 0.9, 0.85, 0.8],
            "rnn.weight_hh_l0": [0.8, 0.7, 0.65, 0.6, 0.55],
            "rnn.weight_ih_l1": [0.6, 0.5, 0.45, 0.4, 0.38],
            "rnn.weight_hh_l1": [0.4, 0.35, 0.3, 0.28, 0.25],
            "fc.weight": [0.3, 0.25, 0.2, 0.18, 0.15],
        },
        training_time=120.5,
        epochs_trained=5,
    )


@pytest.fixture
def gru_history():
    """Create a sample GRU training history."""
    return TrainingHistory(
        train_losses=[0.63, 0.43, 0.33, 0.26, 0.20],
        val_losses=[0.58, 0.40, 0.36, 0.37, 0.38],
        train_accs=[0.64, 0.79, 0.85, 0.89, 0.92],
        val_accs=[0.66, 0.81, 0.84, 0.83, 0.82],
        gradient_norms={
            "embedding.weight": [0.55, 0.45, 0.35, 0.33, 0.30],
            "rnn.weight_ih_l0": [1.1, 0.95, 0.85, 0.80, 0.75],
            "rnn.weight_hh_l0": [0.75, 0.65, 0.60, 0.55, 0.50],
            "rnn.weight_ih_l1": [0.55, 0.48, 0.42, 0.38, 0.35],
            "rnn.weight_hh_l1": [0.38, 0.33, 0.28, 0.25, 0.22],
            "fc.weight": [0.28, 0.22, 0.18, 0.16, 0.13],
        },
        training_time=105.3,
        epochs_trained=5,
    )


@pytest.fixture
def lstm_results():
    """Create sample LSTM evaluation results."""
    return EvaluationResults(
        accuracy=0.852,
        precision=0.861,
        recall=0.840,
        f1=0.850,
        confusion_matrix=[[10600, 1900], [2000, 10500]],
        rnn_type="LSTM",
    )


@pytest.fixture
def gru_results():
    """Create sample GRU evaluation results."""
    return EvaluationResults(
        accuracy=0.845,
        precision=0.853,
        recall=0.834,
        f1=0.843,
        confusion_matrix=[[10500, 2000], [2075, 10425]],
        rnn_type="GRU",
    )


class TestGenerateReflexiones:
    """Tests for the generate_reflexiones function."""

    def test_generates_file(self, lstm_history, gru_history, lstm_results, gru_results):
        """Test that the function creates a REFLEXIONES.md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            assert os.path.exists(result_path)
            assert result_path.endswith("REFLEXIONES.md")

    def test_returns_correct_path(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that the returned path matches expected location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            expected = os.path.join(tmpdir, "REFLEXIONES.md")
            assert result_path == expected

    def test_creates_output_directory(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that the function creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, nested_dir
            )
            assert os.path.exists(result_path)

    def test_contains_all_8_questions(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that the generated file contains all 8 reflection questions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check all 8 section headers exist
            assert "## 1." in content
            assert "## 2." in content
            assert "## 3." in content
            assert "## 4." in content
            assert "## 5." in content
            assert "## 6." in content
            assert "## 7." in content
            assert "## 8." in content

    def test_contains_conclusion(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that the generated file contains a conclusion section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "## Conclusión" in content

    def test_contains_quantitative_lstm_metrics(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that LSTM metrics are embedded in the document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check LSTM accuracy is present
            assert "85.20%" in content
            # Check LSTM F1 is present
            assert "85.00%" in content

    def test_contains_quantitative_gru_metrics(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that GRU metrics are embedded in the document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check GRU accuracy is present
            assert "84.50%" in content
            # Check GRU F1 is present
            assert "84.30%" in content

    def test_contains_training_time_comparison(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that training time comparison is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "120.5" in content
            assert "105.3" in content

    def test_contains_gradient_analysis(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that gradient norm analysis is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Should contain gradient-related content
            assert "gradiente" in content.lower() or "gradient" in content.lower()
            # Should contain per-layer analysis
            assert "media=" in content

    def test_contains_comparison_table(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that a comparison table is present in the document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for markdown table structure
            assert "| Métrica |" in content or "| Métrica|" in content
            assert "| Accuracy |" in content or "| Accuracy|" in content
            assert "| F1-Score |" in content or "| F1-Score|" in content

    def test_written_in_spanish(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that the document is written in Spanish."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for Spanish keywords
            assert "Reflexiones" in content
            assert "Conclusión" in content
            assert "entrenamiento" in content
            assert "rendimiento" in content

    def test_contains_early_stopping_info(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that early stopping information is included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "early stopping" in content.lower()
            assert "patience" in content.lower()

    def test_markdown_title(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """Test that the document starts with the correct title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = generate_reflexiones(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            with open(result_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert content.startswith(
                "# Reflexiones sobre el Experimento de Análisis de Sentimiento"
            )


class TestHelperFunctions:
    """Tests for helper functions in reflexiones module."""

    def test_compute_avg_gradient_norm(self, lstm_history):
        """Test average gradient norm computation."""
        avg = _compute_avg_gradient_norm(lstm_history)
        assert avg > 0
        assert isinstance(avg, float)

    def test_compute_avg_gradient_norm_empty(self):
        """Test average gradient norm with empty history."""
        history = TrainingHistory()
        avg = _compute_avg_gradient_norm(history)
        assert avg == 0.0

    def test_compute_layer_gradient_stats(self, lstm_history):
        """Test per-layer gradient statistics computation."""
        stats = _compute_layer_gradient_stats(lstm_history)
        assert len(stats) > 0
        for _layer_name, layer_stats in stats.items():
            assert "mean" in layer_stats
            assert "max" in layer_stats
            assert "min" in layer_stats
            assert layer_stats["max"] >= layer_stats["mean"] >= layer_stats["min"]

    def test_detect_gradient_issues_none(self, lstm_history):
        """Test that no gradient issues are detected with normal norms."""
        result = _detect_gradient_issues(lstm_history)
        assert result is None

    def test_detect_gradient_issues_vanishing(self):
        """Test detection of vanishing gradients."""
        history = TrainingHistory(
            gradient_norms={
                "rnn.weight_ih_l0": [0.00001, 0.00002, 0.000015],
            }
        )
        result = _detect_gradient_issues(history)
        assert result is not None
        assert "vanishing" in str(result)

    def test_detect_gradient_issues_near_clipping(self):
        """Test detection of near-clipping gradients."""
        history = TrainingHistory(
            gradient_norms={
                "rnn.weight_ih_l0": [4.6, 4.7, 4.8],  # Close to GRAD_CLIP_NORM=5.0
            }
        )
        result = _detect_gradient_issues(history)
        assert result is not None
        assert "near-clipping" in str(result)
