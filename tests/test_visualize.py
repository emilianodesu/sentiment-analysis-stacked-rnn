"""Tests for the visualization module."""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from evaluate import EvaluationResults
from train import TrainingHistory
from visualize import (
    plot_confusion_matrices,
    plot_gradient_norms,
    plot_training_curves,
    save_all_plots,
)


@pytest.fixture
def lstm_history():
    """Create a sample LSTM TrainingHistory."""
    history = TrainingHistory()
    history.train_losses = [0.7, 0.5, 0.4, 0.35, 0.3]
    history.val_losses = [0.65, 0.55, 0.45, 0.42, 0.41]
    history.train_accs = [0.55, 0.7, 0.78, 0.82, 0.85]
    history.val_accs = [0.52, 0.65, 0.72, 0.75, 0.76]
    history.gradient_norms = {
        "embedding.weight": [0.5 + i * 0.01 for i in range(100)],
        "rnn.weight_ih_l0": [1.0 - i * 0.005 for i in range(100)],
        "rnn.weight_hh_l0": [0.8 + i * 0.002 for i in range(100)],
        "fc.weight": [0.3 + i * 0.003 for i in range(100)],
    }
    history.training_time = 120.0
    history.epochs_trained = 5
    return history


@pytest.fixture
def gru_history():
    """Create a sample GRU TrainingHistory."""
    history = TrainingHistory()
    history.train_losses = [0.68, 0.48, 0.38, 0.33, 0.28]
    history.val_losses = [0.63, 0.52, 0.43, 0.40, 0.39]
    history.train_accs = [0.57, 0.72, 0.80, 0.84, 0.87]
    history.val_accs = [0.54, 0.67, 0.74, 0.77, 0.78]
    history.gradient_norms = {
        "embedding.weight": [0.45 + i * 0.01 for i in range(100)],
        "rnn.weight_ih_l0": [0.9 - i * 0.004 for i in range(100)],
        "rnn.weight_hh_l0": [0.7 + i * 0.002 for i in range(100)],
        "fc.weight": [0.25 + i * 0.003 for i in range(100)],
    }
    history.training_time = 100.0
    history.epochs_trained = 5
    return history


@pytest.fixture
def lstm_results():
    """Create sample LSTM EvaluationResults."""
    return EvaluationResults(
        accuracy=0.85,
        precision=0.84,
        recall=0.86,
        f1=0.85,
        confusion_matrix=[[10500, 2000], [1750, 10750]],
        rnn_type="LSTM",
    )


@pytest.fixture
def gru_results():
    """Create sample GRU EvaluationResults."""
    return EvaluationResults(
        accuracy=0.86,
        precision=0.85,
        recall=0.87,
        f1=0.86,
        confusion_matrix=[[10600, 1900], [1600, 10900]],
        rnn_type="GRU",
    )


class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_returns_figure(self, lstm_history, gru_history):
        """plot_training_curves returns a matplotlib Figure."""
        fig = plot_training_curves(lstm_history, gru_history)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_two_subplots(self, lstm_history, gru_history):
        """Figure has exactly 2 subplots (loss and accuracy)."""
        fig = plot_training_curves(lstm_history, gru_history)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_loss_subplot_has_four_lines(self, lstm_history, gru_history):
        """Loss subplot has 4 lines (LSTM train/val + GRU train/val)."""
        fig = plot_training_curves(lstm_history, gru_history)
        ax_loss = fig.get_axes()[0]
        assert len(ax_loss.get_lines()) == 4
        plt.close(fig)

    def test_accuracy_subplot_has_four_lines(self, lstm_history, gru_history):
        """Accuracy subplot has 4 lines (LSTM train/val + GRU train/val)."""
        fig = plot_training_curves(lstm_history, gru_history)
        ax_acc = fig.get_axes()[1]
        assert len(ax_acc.get_lines()) == 4
        plt.close(fig)

    def test_subplots_have_legends(self, lstm_history, gru_history):
        """Both subplots have legends."""
        fig = plot_training_curves(lstm_history, gru_history)
        for ax in fig.get_axes():
            legend = ax.get_legend()
            assert legend is not None
        plt.close(fig)


class TestPlotConfusionMatrices:
    """Tests for plot_confusion_matrices function."""

    def test_returns_figure(self, lstm_results, gru_results):
        """plot_confusion_matrices returns a matplotlib Figure."""
        fig = plot_confusion_matrices(lstm_results, gru_results)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_two_subplots(self, lstm_results, gru_results):
        """Figure has exactly 2 subplots (LSTM and GRU)."""
        fig = plot_confusion_matrices(lstm_results, gru_results)
        axes = fig.get_axes()
        # Seaborn heatmaps add colorbars as extra axes
        # The main axes are the first two
        assert len(axes) >= 2
        plt.close(fig)

    def test_titles_contain_model_names(self, lstm_results, gru_results):
        """Subplot titles contain LSTM and GRU."""
        fig = plot_confusion_matrices(lstm_results, gru_results)
        axes = fig.get_axes()
        titles = [ax.get_title() for ax in axes]
        assert any("LSTM" in t for t in titles)
        assert any("GRU" in t for t in titles)
        plt.close(fig)


class TestPlotGradientNorms:
    """Tests for plot_gradient_norms function."""

    def test_returns_figure(self, lstm_history, gru_history):
        """plot_gradient_norms returns a matplotlib Figure."""
        fig = plot_gradient_norms(lstm_history, gru_history)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_has_two_subplots(self, lstm_history, gru_history):
        """Figure has exactly 2 subplots (LSTM and GRU)."""
        fig = plot_gradient_norms(lstm_history, gru_history)
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_lstm_subplot_has_lines_per_layer(self, lstm_history, gru_history):
        """LSTM subplot has one line per layer in gradient_norms."""
        fig = plot_gradient_norms(lstm_history, gru_history)
        ax_lstm = fig.get_axes()[0]
        num_layers = len(lstm_history.gradient_norms)
        assert len(ax_lstm.get_lines()) == num_layers
        plt.close(fig)

    def test_gru_subplot_has_lines_per_layer(self, lstm_history, gru_history):
        """GRU subplot has one line per layer in gradient_norms."""
        fig = plot_gradient_norms(lstm_history, gru_history)
        ax_gru = fig.get_axes()[1]
        num_layers = len(gru_history.gradient_norms)
        assert len(ax_gru.get_lines()) == num_layers
        plt.close(fig)

    def test_subplots_have_legends(self, lstm_history, gru_history):
        """Both subplots have legends."""
        fig = plot_gradient_norms(lstm_history, gru_history)
        for ax in fig.get_axes():
            legend = ax.get_legend()
            assert legend is not None
        plt.close(fig)


class TestSaveAllPlots:
    """Tests for save_all_plots function."""

    def test_creates_output_directory(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """save_all_plots creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test_outputs")
            save_all_plots(
                lstm_history, gru_history, lstm_results, gru_results, output_dir
            )
            assert os.path.isdir(output_dir)

    def test_saves_training_curves_png(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """save_all_plots creates training_curves.png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_plots(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            assert os.path.isfile(os.path.join(tmpdir, "training_curves.png"))

    def test_saves_confusion_matrices_png(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """save_all_plots creates confusion_matrices.png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_plots(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            assert os.path.isfile(os.path.join(tmpdir, "confusion_matrices.png"))

    def test_saves_gradient_norms_png(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """save_all_plots creates gradient_norms.png."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_plots(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            assert os.path.isfile(os.path.join(tmpdir, "gradient_norms.png"))

    def test_png_files_are_non_empty(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """All saved PNG files have non-zero size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_plots(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )
            for filename in [
                "training_curves.png",
                "confusion_matrices.png",
                "gradient_norms.png",
            ]:
                filepath = os.path.join(tmpdir, filename)
                assert os.path.getsize(filepath) > 0

    def test_closes_figures_after_saving(
        self, lstm_history, gru_history, lstm_results, gru_results
    ):
        """save_all_plots closes all figures to free memory."""
        # Record open figures before
        plt.close("all")
        open_before = len(plt.get_fignums())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_plots(
                lstm_history, gru_history, lstm_results, gru_results, tmpdir
            )

        # After save_all_plots, no new figures should remain open
        open_after = len(plt.get_fignums())
        assert open_after == open_before
