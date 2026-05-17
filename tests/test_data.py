"""Tests for data.py: clean_text and download_imdb_dataset.

Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6
"""

import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data import clean_text, download_imdb_dataset


# ─── Unit Tests for clean_text ────────────────────────────────────────────────


class TestCleanTextHTMLRemoval:
    """Validate that clean_text removes all HTML tags."""

    def test_removes_br_tags(self):
        text = "Hello<br />world"
        result = clean_text(text)
        assert "<br" not in result
        assert "hello" in result
        assert "world" in result

    def test_removes_paragraph_tags(self):
        text = "<p>This is a paragraph.</p>"
        result = clean_text(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "this is a paragraph" in result

    def test_removes_nested_html(self):
        text = "<div><span>nested</span></div>"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result
        assert "nested" in result

    def test_removes_html_with_attributes(self):
        text = '<a href="http://example.com">link</a>'
        result = clean_text(text)
        assert "<a" not in result
        assert "href" not in result
        assert "link" in result


class TestCleanTextLowercase:
    """Validate that clean_text converts all text to lowercase."""

    def test_uppercase_to_lowercase(self):
        text = "HELLO WORLD"
        result = clean_text(text)
        assert result == "hello world"

    def test_mixed_case_to_lowercase(self):
        text = "HeLLo WoRLd"
        result = clean_text(text)
        assert result == "hello world"

    def test_already_lowercase(self):
        text = "hello world"
        result = clean_text(text)
        assert result == "hello world"


class TestCleanTextPunctuation:
    """Validate that clean_text removes punctuation and non-alphanumeric characters."""

    def test_removes_periods_and_commas(self):
        text = "Hello, world. How are you?"
        result = clean_text(text)
        assert "," not in result
        assert "." not in result
        assert "?" not in result

    def test_removes_special_characters(self):
        text = "price: $100! great@deal #1"
        result = clean_text(text)
        assert "$" not in result
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_removes_numbers(self):
        text = "I have 3 cats and 2 dogs"
        result = clean_text(text)
        assert "3" not in result
        assert "2" not in result
        assert "i have" in result
        assert "cats and" in result

    def test_keeps_only_letters_and_spaces(self):
        text = "abc!@#123 def"
        result = clean_text(text)
        # Only letters remain
        for char in result:
            assert char.isalpha() or char == ' '


class TestCleanTextTokenization:
    """Validate that clean_text tokenizes by space-based splitting."""

    def test_normalizes_multiple_spaces(self):
        text = "hello    world"
        result = clean_text(text)
        assert result == "hello world"

    def test_strips_leading_trailing_spaces(self):
        text = "  hello world  "
        result = clean_text(text)
        assert result == "hello world"

    def test_empty_string_returns_empty(self):
        result = clean_text("")
        assert result == ""

    def test_only_html_returns_empty(self):
        result = clean_text("<br /><br />")
        assert result == ""

    def test_full_pipeline(self):
        """Test the complete cleaning pipeline on a realistic review."""
        text = "<br />This is a GREAT movie!!! I loved it 100%.<br />"
        result = clean_text(text)
        assert result == "this is a great movie i loved it"


# ─── Property-Based Tests for clean_text ──────────────────────────────────────


class TestCleanTextProperties:
    """Property-based tests for clean_text.

    **Validates: Requirements 1.2, 1.3, 1.4, 1.5**
    """

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_output_contains_only_lowercase_letters_and_spaces(self, text):
        """After cleaning, output contains only lowercase letters and single spaces."""
        result = clean_text(text)
        for char in result:
            assert char.isalpha() or char == ' ', (
                f"Unexpected character '{char}' in output"
            )
        # No double spaces
        assert "  " not in result

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_output_is_lowercase(self, text):
        """All alphabetic characters in output are lowercase."""
        result = clean_text(text)
        assert result == result.lower()

    @given(text=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'Zs')),
        min_size=1, max_size=200
    ))
    @settings(max_examples=50)
    def test_no_html_tags_in_output(self, text):
        """Output never contains HTML angle brackets."""
        # Inject some HTML into the input
        html_text = f"<div>{text}</div>"
        result = clean_text(html_text)
        assert "<" not in result
        assert ">" not in result

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_output_has_no_leading_or_trailing_spaces(self, text):
        """Output is stripped of leading/trailing whitespace."""
        result = clean_text(text)
        if result:  # non-empty
            assert result[0] != ' '
            assert result[-1] != ' '


# ─── Tests for download_imdb_dataset error handling ───────────────────────────


class TestDownloadIMDBDataset:
    """Validate error handling for dataset download.

    **Validates: Requirements 1.1, 1.6**
    """

    def test_raises_runtime_error_on_failure(self, monkeypatch):
        """If download fails, a RuntimeError with informative message is raised."""
        def mock_load_dataset(*args, **kwargs):
            raise ConnectionError("Network unreachable")

        # Ensure data module is loaded so monkeypatch can target it
        import data  # noqa: F811
        monkeypatch.setattr(data, "load_dataset", mock_load_dataset, raising=False)

        # We need to patch it where it's imported
        import datasets  # noqa: F811
        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        with pytest.raises(RuntimeError, match="Failed to download the IMDB dataset"):
            download_imdb_dataset()

    def test_error_message_includes_reason(self, monkeypatch):
        """The error message includes the original failure reason."""
        def mock_load_dataset(*args, **kwargs):
            raise ConnectionError("DNS resolution failed")

        import datasets  # noqa: F811
        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        with pytest.raises(RuntimeError, match="DNS resolution failed"):
            download_imdb_dataset()


# ─── Import Vocabulary ────────────────────────────────────────────────────────

from data import Vocabulary


# ─── Unit Tests for Vocabulary ────────────────────────────────────────────────


class TestVocabularyConstruction:
    """Validate Vocabulary class construction and special tokens.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
    """

    def test_pad_is_index_zero(self):
        """PAD token must always be at index 0."""
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build([["hello", "world", "hello", "world"]])
        assert vocab.word2idx["<PAD>"] == 0

    def test_unk_is_index_one(self):
        """UNK token must always be at index 1."""
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build([["hello", "world", "hello", "world"]])
        assert vocab.word2idx["<UNK>"] == 1

    def test_min_freq_filtering(self):
        """Words appearing fewer than min_freq times are excluded."""
        texts = [["hello", "hello", "world", "world", "rare"]]
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(texts)
        assert "hello" in vocab.word2idx
        assert "world" in vocab.word2idx
        assert "rare" not in vocab.word2idx

    def test_max_size_cap(self):
        """Vocabulary size is capped at max_size + 2 (PAD + UNK)."""
        # Create many unique words that all appear >= 2 times
        words = [f"word{i}" for i in range(100)]
        texts = [words + words]  # Each word appears exactly 2 times
        vocab = Vocabulary(min_freq=2, max_size=10)
        vocab.build(texts)
        # max_size=10 words + PAD + UNK = 12 total
        assert len(vocab.word2idx) == 12

    def test_bidirectional_mapping_consistency(self):
        """word2idx and idx2word are consistent inverses."""
        texts = [["the", "movie", "is", "great", "the", "movie", "is", "great"]]
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(texts)
        for word, idx in vocab.word2idx.items():
            assert vocab.idx2word[idx] == word
        for idx, word in vocab.idx2word.items():
            assert vocab.word2idx[word] == idx

    def test_encode_known_words(self):
        """Known words are encoded to their correct indices."""
        texts = [["hello", "world", "hello", "world"]]
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(texts)
        encoded = vocab.encode(["hello", "world"])
        assert encoded == [vocab.word2idx["hello"], vocab.word2idx["world"]]

    def test_encode_unknown_words_to_unk(self):
        """Unknown words are encoded to UNK index (1)."""
        texts = [["hello", "world", "hello", "world"]]
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(texts)
        encoded = vocab.encode(["unknown", "missing"])
        assert encoded == [1, 1]

    def test_encode_mixed_known_unknown(self):
        """Mix of known and unknown words encodes correctly."""
        texts = [["hello", "world", "hello", "world"]]
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(texts)
        encoded = vocab.encode(["hello", "unknown", "world"])
        assert encoded[0] == vocab.word2idx["hello"]
        assert encoded[1] == 1  # UNK
        assert encoded[2] == vocab.word2idx["world"]

    def test_empty_input_builds_only_special_tokens(self):
        """Building from empty input produces only PAD and UNK."""
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build([])
        assert len(vocab.word2idx) == 2
        assert vocab.word2idx["<PAD>"] == 0
        assert vocab.word2idx["<UNK>"] == 1

    def test_words_sorted_by_frequency(self):
        """More frequent words get lower indices (after PAD and UNK)."""
        texts = [["a", "a", "a", "b", "b", "b", "b", "c", "c"]]
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(texts)
        # b (4 times) should have lower index than a (3 times), a lower than c (2 times)
        assert vocab.word2idx["b"] < vocab.word2idx["a"]
        assert vocab.word2idx["a"] < vocab.word2idx["c"]


# ─── Property-Based Tests for Vocabulary (Property 1) ─────────────────────────


class TestVocabularyProperties:
    """Property-based tests for Vocabulary integrity.

    **Validates: Requirements 2.4, 2.5**
    """

    @given(
        word_lists=st.lists(
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=('Ll',)),
                    min_size=1,
                    max_size=10
                ),
                min_size=0,
                max_size=50
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_vocab_size_within_bounds(self, word_lists):
        """len(vocab.word2idx) <= VOCAB_SIZE + 2 (including PAD and UNK).

        **Validates: Requirements 2.3**
        """
        max_size = 100  # Use small max_size for testing
        vocab = Vocabulary(min_freq=2, max_size=max_size)
        vocab.build(word_lists)
        assert len(vocab.word2idx) <= max_size + 2

    @given(
        word_lists=st.lists(
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=('Ll',)),
                    min_size=1,
                    max_size=10
                ),
                min_size=0,
                max_size=50
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_pad_always_zero(self, word_lists):
        """vocab.word2idx["<PAD>"] == 0 always.

        **Validates: Requirements 2.4**
        """
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(word_lists)
        assert vocab.word2idx["<PAD>"] == 0

    @given(
        word_lists=st.lists(
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=('Ll',)),
                    min_size=1,
                    max_size=10
                ),
                min_size=0,
                max_size=50
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_unk_always_one(self, word_lists):
        """vocab.word2idx["<UNK>"] == 1 always.

        **Validates: Requirements 2.4**
        """
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(word_lists)
        assert vocab.word2idx["<UNK>"] == 1

    @given(
        word_lists=st.lists(
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=('Ll',)),
                    min_size=1,
                    max_size=10
                ),
                min_size=0,
                max_size=50
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100)
    def test_bidirectional_mapping_roundtrip(self, word_lists):
        """For all words w in vocabulary: vocab.idx2word[vocab.word2idx[w]] == w.

        **Validates: Requirements 2.5**
        """
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(word_lists)
        for word, idx in vocab.word2idx.items():
            assert vocab.idx2word[idx] == word
        for idx, word in vocab.idx2word.items():
            assert vocab.word2idx[word] == idx

    @given(
        word_lists=st.lists(
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=('Ll',)),
                    min_size=1,
                    max_size=10
                ),
                min_size=1,
                max_size=50
            ),
            min_size=1,
            max_size=20
        ),
        unknown_words=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=('Lu',)),
                min_size=1,
                max_size=10
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=50)
    def test_unknown_words_encode_to_unk(self, word_lists, unknown_words):
        """OOV words always map to UNK index (1).

        **Validates: Requirements 2.6**
        """
        vocab = Vocabulary(min_freq=2, max_size=25_000)
        vocab.build(word_lists)
        encoded = vocab.encode(unknown_words)
        for idx in encoded:
            assert idx == 1 or unknown_words[encoded.index(idx)] in vocab.word2idx



# ─── Import IMDBDataset ───────────────────────────────────────────────────────

import torch
from data import IMDBDataset


# ─── Unit Tests for IMDBDataset ───────────────────────────────────────────────


class TestIMDBDataset:
    """Validate IMDBDataset class behavior.

    **Validates: Requirements 3.6**
    """

    def test_len_returns_correct_count(self):
        """__len__ returns the number of samples."""
        encodings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        labels = [0, 1, 1]
        dataset = IMDBDataset(encodings, labels)
        assert len(dataset) == 3

    def test_getitem_returns_tuple(self):
        """__getitem__ returns a tuple of (sequence, label) tensors."""
        encodings = [[1, 2, 3], [4, 5, 6]]
        labels = [0, 1]
        dataset = IMDBDataset(encodings, labels)
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_sequence_tensor_is_long(self):
        """Sequences are stored as LongTensor."""
        encodings = [[1, 2, 3]]
        labels = [1]
        dataset = IMDBDataset(encodings, labels)
        seq, _ = dataset[0]
        assert seq.dtype == torch.long

    def test_label_tensor_is_float(self):
        """Labels are stored as FloatTensor."""
        encodings = [[1, 2, 3]]
        labels = [1]
        dataset = IMDBDataset(encodings, labels)
        _, label = dataset[0]
        assert label.dtype == torch.float32

    def test_getitem_correct_values(self):
        """__getitem__ returns correct values for a given index."""
        encodings = [[10, 20, 30], [40, 50, 60]]
        labels = [0, 1]
        dataset = IMDBDataset(encodings, labels)
        seq, label = dataset[1]
        assert torch.equal(seq, torch.tensor([40, 50, 60], dtype=torch.long))
        assert label.item() == 1.0

    def test_empty_dataset(self):
        """Dataset with no samples has length 0."""
        dataset = IMDBDataset([], [])
        assert len(dataset) == 0

    def test_single_sample(self):
        """Dataset with one sample works correctly."""
        encodings = [[0, 0, 1, 2, 3]]
        labels = [0]
        dataset = IMDBDataset(encodings, labels)
        assert len(dataset) == 1
        seq, label = dataset[0]
        assert seq.shape == (5,)
        assert label.item() == 0.0

    def test_padded_sequences_shape(self):
        """All sequences in the dataset have the same length (padded)."""
        # Simulate padded sequences of length 200
        encodings = [[i] * 200 for i in range(5)]
        labels = [0, 1, 0, 1, 0]
        dataset = IMDBDataset(encodings, labels)
        for i in range(len(dataset)):
            seq, _ = dataset[i]
            assert seq.shape == (200,)


# ─── Import get_dataloaders ──────────────────────────────────────────────────

from unittest.mock import patch
from torch.utils.data import DataLoader
from data import get_dataloaders


# ─── Unit Tests for get_dataloaders ──────────────────────────────────────────


class TestGetDataloaders:
    """Validate get_dataloaders pipeline orchestration.

    **Validates: Requirements 3.3, 3.4, 3.5**
    """

    @pytest.fixture
    def mock_imdb_dataset(self):
        """Create a small mock IMDB dataset for testing."""
        import random
        rng = random.Random(123)
        words_pool = [
            "amazing", "terrible", "wonderful", "boring", "exciting",
            "dull", "brilliant", "awful", "fantastic", "horrible",
            "superb", "mediocre", "outstanding", "disappointing", "thrilling",
            "tedious", "captivating", "predictable", "engaging", "forgettable",
            "masterpiece", "disaster", "gem", "waste", "triumph",
        ]

        def make_review(idx):
            # Generate unique reviews with random word combinations
            n_words = rng.randint(5, 15)
            chosen = [rng.choice(words_pool) for _ in range(n_words)]
            return " ".join(chosen) + f" review {idx}"

        train_samples = [
            {"text": make_review(i), "label": i % 2}
            for i in range(100)
        ]
        test_samples = [
            {"text": make_review(i + 1000), "label": i % 2}
            for i in range(50)
        ]

        class FakeDatasetSplit:
            def __init__(self, data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        class FakeDatasetDict:
            def __init__(self, train, test):
                self._splits = {"train": FakeDatasetSplit(train), "test": FakeDatasetSplit(test)}

            def __getitem__(self, key):
                return self._splits[key]

        return FakeDatasetDict(train_samples, test_samples)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_returns_correct_tuple_structure(self, mock_config, mock_download, mock_imdb_dataset):
        """get_dataloaders returns (train_loader, val_loader, test_loader, vocab)."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        result = get_dataloaders(batch_size=16, seed=42)

        assert len(result) == 4
        train_loader, val_loader, test_loader, vocab = result
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert isinstance(vocab, Vocabulary)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_split_sizes_correct(self, mock_config, mock_download, mock_imdb_dataset):
        """Train has TRAIN_SIZE samples, val has VAL_SIZE, test has full test set."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=16, seed=42)

        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 20
        assert len(test_loader.dataset) == 50

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_reproducibility_with_same_seed(self, mock_config, mock_download, mock_imdb_dataset):
        """Same seed produces identical train/val splits."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        _, val_loader1, _, _ = get_dataloaders(batch_size=16, seed=42)
        mock_download.return_value = mock_imdb_dataset
        _, val_loader2, _, _ = get_dataloaders(batch_size=16, seed=42)

        # Same seed → same validation data
        for (seq1, lbl1), (seq2, lbl2) in zip(val_loader1, val_loader2):
            assert torch.equal(seq1, seq2)
            assert torch.equal(lbl1, lbl2)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_different_seed_produces_different_split(self, mock_config, mock_download, mock_imdb_dataset):
        """Different seeds produce different train/val splits."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        _, val_loader1, _, _ = get_dataloaders(batch_size=100, seed=42)
        mock_download.return_value = mock_imdb_dataset
        _, val_loader2, _, _ = get_dataloaders(batch_size=100, seed=99)

        # Different seeds should produce different validation sets
        val_batch1 = next(iter(val_loader1))
        val_batch2 = next(iter(val_loader2))
        # At least the sequences should differ (labels might coincidentally match)
        assert not torch.equal(val_batch1[0], val_batch2[0])

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_sequences_are_padded_to_max_len(self, mock_config, mock_download, mock_imdb_dataset):
        """All sequences in all loaders have length MAX_LEN."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=16, seed=42)

        # Check first batch of each loader
        for loader in [train_loader, val_loader, test_loader]:
            batch_seq, _ = next(iter(loader))
            assert batch_seq.shape[1] == 200

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_vocab_built_from_train_only(self, mock_config, mock_download, mock_imdb_dataset):
        """Vocabulary is built from training split texts only."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        _, _, _, vocab = get_dataloaders(batch_size=16, seed=42)

        # Vocab should have PAD and UNK at minimum
        assert vocab.word2idx["<PAD>"] == 0
        assert vocab.word2idx["<UNK>"] == 1
        # Should have some words from the training data
        assert len(vocab.word2idx) > 2

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_train_loader_shuffles(self, mock_config, mock_download, mock_imdb_dataset):
        """Train DataLoader has shuffle=True (different batch order across epochs)."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        train_loader, _, _, _ = get_dataloaders(batch_size=16, seed=42)

        # Collect indices from two full epochs by concatenating all sequences
        epoch1_seqs = torch.cat([seq for seq, _ in train_loader], dim=0)
        epoch2_seqs = torch.cat([seq for seq, _ in train_loader], dim=0)

        # With shuffle=True and 80 diverse samples, the order should differ
        assert not torch.equal(epoch1_seqs, epoch2_seqs), "Train loader should shuffle between epochs"

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_batch_size_respected(self, mock_config, mock_download, mock_imdb_dataset):
        """DataLoaders use the specified batch_size."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        train_loader, _, _, _ = get_dataloaders(batch_size=32, seed=42)

        batch_seq, batch_lbl = next(iter(train_loader))
        assert batch_seq.shape[0] == 32
        assert batch_lbl.shape[0] == 32


# ─── Import pad_sequence ─────────────────────────────────────────────────────

from data import pad_sequence


# ─── Property-Based Tests for pad_sequence (Property 2) ──────────────────────


class TestPadSequenceProperties:
    """Property-based tests for pad_sequence invariants.

    **Validates: Requirements 3.1, 3.2**
    """

    @given(
        seq=st.lists(
            st.integers(min_value=0, max_value=25001),
            min_size=0,
            max_size=500
        ),
        max_len=st.integers(min_value=1, max_value=300)
    )
    @settings(max_examples=200)
    def test_output_length_always_equals_max_len(self, seq, max_len):
        """pad_sequence always produces output of exactly max_len length.

        **Validates: Requirements 3.1, 3.2**
        """
        result = pad_sequence(seq, max_len)
        assert len(result) == max_len

    @given(
        seq=st.lists(
            st.integers(min_value=0, max_value=25001),
            min_size=0,
            max_size=500
        ),
        max_len=st.integers(min_value=1, max_value=300)
    )
    @settings(max_examples=200)
    def test_all_values_are_non_negative_integers(self, seq, max_len):
        """All values in padded output are non-negative integers.

        **Validates: Requirements 3.1, 3.2**
        """
        result = pad_sequence(seq, max_len)
        for val in result:
            assert isinstance(val, int)
            assert val >= 0

    @given(
        max_len=st.integers(min_value=1, max_value=200),
        extra=st.integers(min_value=1, max_value=300)
    )
    @settings(max_examples=200)
    def test_truncation_preserves_first_max_len_elements(self, max_len, extra):
        """When sequence is longer than max_len, first max_len elements are kept.

        **Validates: Requirements 3.1**
        """
        # Build a sequence guaranteed to be longer than max_len
        seq = list(range(1, max_len + extra + 1))
        result = pad_sequence(seq, max_len)
        assert result == seq[:max_len]

    @given(
        max_len=st.integers(min_value=2, max_value=300),
        seq_len=st.integers(min_value=0, max_value=299)
    )
    @settings(max_examples=200)
    def test_padding_uses_only_zeros(self, max_len, seq_len):
        """Padding uses only zeros (PAD=0) for the added elements.

        **Validates: Requirements 3.2**
        """
        from hypothesis import assume
        assume(seq_len < max_len)
        # Build a sequence of non-zero values shorter than max_len
        seq = list(range(1, seq_len + 1))
        result = pad_sequence(seq, max_len)
        # Original elements are preserved at the start
        assert result[:seq_len] == seq
        # Padded elements are all zeros
        padding = result[seq_len:]
        assert all(v == 0 for v in padding)

    @given(
        seq=st.lists(
            st.integers(min_value=0, max_value=25001),
            min_size=0,
            max_size=500
        )
    )
    @settings(max_examples=100)
    def test_output_length_invariant_with_fixed_max_len(self, seq):
        """For any input sequence, output length == MAX_LEN (200) invariant.

        **Validates: Requirements 3.1, 3.2**
        """
        result = pad_sequence(seq, 200)
        assert len(result) == 200


# ─── Unit Tests for DataLoader Configuration ─────────────────────────────────


class TestDataLoaderConfiguration:
    """Validate DataLoader shuffle, batch size, and tensor dtype configuration.

    **Validates: Requirements 3.5**
    """

    @pytest.fixture
    def mock_imdb_dataset(self):
        """Create a small mock IMDB dataset for testing."""
        import random
        rng = random.Random(456)
        words_pool = [
            "amazing", "terrible", "wonderful", "boring", "exciting",
            "dull", "brilliant", "awful", "fantastic", "horrible",
            "superb", "mediocre", "outstanding", "disappointing",
            "thrilling", "tedious", "captivating", "predictable",
            "engaging", "forgettable", "masterpiece", "disaster",
            "gem", "waste", "triumph",
        ]

        def make_review(idx):
            n_words = rng.randint(5, 15)
            chosen = [rng.choice(words_pool) for _ in range(n_words)]
            return " ".join(chosen) + f" review {idx}"

        train_samples = [
            {"text": make_review(i), "label": i % 2}
            for i in range(100)
        ]
        test_samples = [
            {"text": make_review(i + 1000), "label": i % 2}
            for i in range(50)
        ]

        class FakeDatasetSplit:
            def __init__(self, data):
                self._data = data

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        class FakeDatasetDict:
            def __init__(self, train, test):
                self._splits = {
                    "train": FakeDatasetSplit(train),
                    "test": FakeDatasetSplit(test),
                }

            def __getitem__(self, key):
                return self._splits[key]

        return FakeDatasetDict(train_samples, test_samples)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_train_loader_has_shuffle_true(
        self, mock_config, mock_download, mock_imdb_dataset
    ):
        """Train DataLoader has shuffle=True (order differs across epochs)."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        train_loader, _, _, _ = get_dataloaders(batch_size=16, seed=42)

        # Two full iterations should produce different orderings
        epoch1 = torch.cat([s for s, _ in train_loader], dim=0)
        epoch2 = torch.cat([s for s, _ in train_loader], dim=0)
        assert not torch.equal(epoch1, epoch2)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_val_loader_has_shuffle_false(
        self, mock_config, mock_download, mock_imdb_dataset
    ):
        """Val DataLoader has shuffle=False (same order every iteration)."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        _, val_loader, _, _ = get_dataloaders(batch_size=16, seed=42)

        epoch1 = torch.cat([s for s, _ in val_loader], dim=0)
        epoch2 = torch.cat([s for s, _ in val_loader], dim=0)
        assert torch.equal(epoch1, epoch2)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_test_loader_has_shuffle_false(
        self, mock_config, mock_download, mock_imdb_dataset
    ):
        """Test DataLoader has shuffle=False (same order every iteration)."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        _, _, test_loader, _ = get_dataloaders(batch_size=16, seed=42)

        epoch1 = torch.cat([s for s, _ in test_loader], dim=0)
        epoch2 = torch.cat([s for s, _ in test_loader], dim=0)
        assert torch.equal(epoch1, epoch2)

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_batch_size_correctly_applied(
        self, mock_config, mock_download, mock_imdb_dataset
    ):
        """Batch size is correctly applied to all loaders."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        batch_size = 16
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            batch_size=batch_size, seed=42
        )

        # First batch of each loader should have batch_size samples
        train_seq, train_lbl = next(iter(train_loader))
        assert train_seq.shape[0] == batch_size
        assert train_lbl.shape[0] == batch_size

        val_seq, val_lbl = next(iter(val_loader))
        assert val_seq.shape[0] == batch_size
        assert val_lbl.shape[0] == batch_size

        test_seq, test_lbl = next(iter(test_loader))
        assert test_seq.shape[0] == batch_size
        assert test_lbl.shape[0] == batch_size

    @patch("data.download_imdb_dataset")
    @patch("data.config")
    def test_tensor_dtypes_correct(
        self, mock_config, mock_download, mock_imdb_dataset
    ):
        """Sequences are LongTensor and labels are FloatTensor."""
        mock_download.return_value = mock_imdb_dataset
        mock_config.TRAIN_SIZE = 80
        mock_config.VAL_SIZE = 20
        mock_config.MIN_FREQ = 2
        mock_config.VOCAB_SIZE = 25_000
        mock_config.MAX_LEN = 200

        train_loader, val_loader, test_loader, _ = get_dataloaders(
            batch_size=16, seed=42
        )

        for loader in [train_loader, val_loader, test_loader]:
            seq, lbl = next(iter(loader))
            assert seq.dtype == torch.long, (
                f"Sequences should be LongTensor, got {seq.dtype}"
            )
            assert lbl.dtype == torch.float32, (
                f"Labels should be FloatTensor, got {lbl.dtype}"
            )
