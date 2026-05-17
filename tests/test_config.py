"""Tests that validate config values and project structure.

Validates: Requirements 9.1, 9.5
"""

import os
import sys

import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import config


# ─── Config Value Tests ───────────────────────────────────────────────────────


class TestConfigValues:
    """Validate all config values exist and have correct types and values."""

    def test_seed(self):
        assert isinstance(config.SEED, int)
        assert config.SEED == 42

    def test_vocab_size(self):
        assert isinstance(config.VOCAB_SIZE, int)
        assert config.VOCAB_SIZE == 25000

    def test_min_freq(self):
        assert isinstance(config.MIN_FREQ, int)
        assert config.MIN_FREQ == 2

    def test_max_len(self):
        assert isinstance(config.MAX_LEN, int)
        assert config.MAX_LEN == 200

    def test_embedding_dim(self):
        assert isinstance(config.EMBEDDING_DIM, int)
        assert config.EMBEDDING_DIM == 128

    def test_hidden_size(self):
        assert isinstance(config.HIDDEN_SIZE, int)
        assert config.HIDDEN_SIZE == 256

    def test_num_layers(self):
        assert isinstance(config.NUM_LAYERS, int)
        assert config.NUM_LAYERS == 2

    def test_rnn_dropout(self):
        assert isinstance(config.RNN_DROPOUT, float)
        assert config.RNN_DROPOUT == 0.3

    def test_embed_dropout(self):
        assert isinstance(config.EMBED_DROPOUT, float)
        assert config.EMBED_DROPOUT == 0.5

    def test_batch_size(self):
        assert isinstance(config.BATCH_SIZE, int)
        assert config.BATCH_SIZE == 64

    def test_learning_rate(self):
        assert isinstance(config.LEARNING_RATE, float)
        assert config.LEARNING_RATE == 0.001

    def test_max_epochs(self):
        assert isinstance(config.MAX_EPOCHS, int)
        assert config.MAX_EPOCHS == 10

    def test_patience(self):
        assert isinstance(config.PATIENCE, int)
        assert config.PATIENCE == 5

    def test_grad_clip_norm(self):
        assert isinstance(config.GRAD_CLIP_NORM, float)
        assert config.GRAD_CLIP_NORM == 5.0

    def test_train_size(self):
        assert isinstance(config.TRAIN_SIZE, int)
        assert config.TRAIN_SIZE == 20000

    def test_val_size(self):
        assert isinstance(config.VAL_SIZE, int)
        assert config.VAL_SIZE == 5000

    def test_gradient_monitor_steps(self):
        assert isinstance(config.GRADIENT_MONITOR_STEPS, int)
        assert config.GRADIENT_MONITOR_STEPS == 100

    def test_output_dir(self):
        assert isinstance(config.OUTPUT_DIR, str)
        assert config.OUTPUT_DIR == "outputs/"

    def test_checkpoint_dir(self):
        assert isinstance(config.CHECKPOINT_DIR, str)
        assert config.CHECKPOINT_DIR == "checkpoints/"


# ─── Project Structure Tests ──────────────────────────────────────────────────


class TestProjectStructure:
    """Validate that all required project files and directories exist."""

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @pytest.mark.parametrize(
        "module_file",
        [
            "config.py",
            "data.py",
            "model.py",
            "train.py",
            "evaluate.py",
            "visualize.py",
            "reflexiones.py",
            "main.py",
        ],
    )
    def test_module_files_exist(self, module_file):
        path = os.path.join(self.PROJECT_ROOT, "src", module_file)
        assert os.path.isfile(path), f"Module file missing: src/{module_file}"

    @pytest.mark.parametrize(
        "directory",
        [
            "outputs",
            "checkpoints",
            "tests",
        ],
    )
    def test_directories_exist(self, directory):
        path = os.path.join(self.PROJECT_ROOT, directory)
        assert os.path.isdir(path), f"Directory missing: {directory}"

    def test_requirements_txt_exists(self):
        path = os.path.join(self.PROJECT_ROOT, "requirements.txt")
        assert os.path.isfile(path), "requirements.txt is missing"
