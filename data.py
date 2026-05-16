"""Data acquisition, cleaning, vocabulary, and DataLoader utilities."""

import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

import config


def clean_text(text: str) -> str:
    """Clean a review text by removing HTML, lowercasing, removing punctuation, and tokenizing.

    Steps:
    1. Remove HTML tags (e.g., <br />, <p>, etc.) using regex
    2. Convert to lowercase
    3. Remove punctuation and non-alphanumeric characters (keep only letters and spaces)
    4. Tokenize by splitting on spaces (remove empty tokens from multiple spaces)

    Returns the cleaned text as a single string (space-separated tokens).
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove non-alpha characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize by splitting on whitespace and rejoin to normalize spacing
    tokens = text.split()
    return ' '.join(tokens)


def download_imdb_dataset():
    """Download the IMDB dataset using Hugging Face datasets library.

    Returns a DatasetDict with 'train' and 'test' splits, each containing
    25,000 reviews with 'text' and 'label' fields.

    Raises:
        RuntimeError: If the download fails and no cached version is available,
            with an informative message including the failure reason.

    Satisfies: REQ-1.1, REQ-1.6
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' library is required but not installed. "
            "Install it with: pip install datasets"
        ) from e

    try:
        dataset = load_dataset("imdb")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download the IMDB dataset. "
            f"Ensure you have an internet connection or a previously cached copy. "
            f"Reason: {e}"
        ) from e

    return dataset


def pad_sequence(encoded: list[int], max_len: int) -> list[int]:
    """Pad or truncate an encoded sequence to exactly max_len.

    If the sequence is longer than max_len, truncate it.
    If the sequence is shorter than max_len, pad with zeros (PAD=0) at the end.

    Args:
        encoded: List of integer-encoded token indices.
        max_len: Target length for the output sequence.

    Returns:
        A list of exactly max_len integers.

    Satisfies: REQ-3.1, REQ-3.2
    """
    if len(encoded) > max_len:
        return encoded[:max_len]
    elif len(encoded) < max_len:
        return encoded + [0] * (max_len - len(encoded))
    else:
        return list(encoded)


class Vocabulary:
    """Token-to-index vocabulary with special tokens PAD and UNK.

    Builds a vocabulary from tokenized training texts, filtering by minimum
    frequency and capping at a maximum size. Provides bidirectional mappings
    between words and indices.

    Satisfies: REQ-2.1 to REQ-2.6
    """

    def __init__(self, min_freq: int = 2, max_size: int = 25_000):
        """Initialize vocabulary with filtering parameters.

        Args:
            min_freq: Minimum frequency a word must have to be included.
            max_size: Maximum number of words (excluding PAD and UNK).
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def build(self, tokenized_texts: list[list[str]]) -> None:
        """Build vocabulary from tokenized training texts.

        Counts word frequencies across all texts, filters by min_freq,
        keeps only the top max_size most frequent words, and assigns
        indices starting after the special tokens (PAD=0, UNK=1).

        Args:
            tokenized_texts: List of tokenized reviews (each is a list of strings).
        """
        # Count word frequencies across all training texts
        freq = Counter()
        for tokens in tokenized_texts:
            freq.update(tokens)

        # Filter by min_freq and sort by frequency (descending), then alphabetically for ties
        filtered = [
            (word, count) for word, count in freq.items() if count >= self.min_freq
        ]
        filtered.sort(key=lambda x: (-x[1], x[0]))

        # Cap at max_size
        filtered = filtered[:self.max_size]

        # Build mappings with mandatory special tokens
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        for idx, (word, _) in enumerate(filtered, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens: list[str]) -> list[int]:
        """Encode a list of tokens into their corresponding indices.

        Words not in the vocabulary are mapped to UNK (index 1).

        Args:
            tokens: List of string tokens to encode.

        Returns:
            List of integer indices.
        """
        unk_idx = self.word2idx["<UNK>"]
        return [self.word2idx.get(token, unk_idx) for token in tokens]


class IMDBDataset(Dataset):
    """PyTorch Dataset for IMDB encoded sequences and labels.

    Stores padded integer sequences as LongTensor and binary labels as FloatTensor.
    Provides standard Dataset interface for use with DataLoader.

    Satisfies: REQ-3.6
    """

    def __init__(self, encodings: list[list[int]], labels: list[int]):
        """Initialize dataset with encoded sequences and labels.

        Args:
            encodings: List of padded integer sequences (each of length MAX_LEN).
            labels: List of binary labels (0 or 1).
        """
        self.encodings = torch.tensor(encodings, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a (sequence_tensor, label_tensor) pair for the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (sequence LongTensor, label FloatTensor).
        """
        return self.encodings[idx], self.labels[idx]


def get_dataloaders(
    batch_size: int = 64, seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader, "Vocabulary"]:
    """Orchestrate the full data pipeline: download, clean, vocab, encode, pad, split, DataLoaders.

    Steps:
    1. Download the IMDB dataset (25K train, 25K test)
    2. Clean all texts using clean_text()
    3. Build vocabulary from training texts only (20K split)
    4. Encode all texts using the vocabulary
    5. Pad all sequences to MAX_LEN
    6. Split the 25K training set into 20K train + 5K validation (seeded)
    7. Create IMDBDataset instances for train, val, and test
    8. Return DataLoaders with specified batch_size

    Args:
        batch_size: Number of samples per batch (default: 64).
        seed: Random seed for reproducible train/val split (default: 42).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocabulary).

    Satisfies: REQ-1.1, REQ-3.3, REQ-3.4, REQ-3.5
    """
    # Step 1: Download dataset
    dataset = download_imdb_dataset()

    # Step 2: Clean all texts
    train_texts_clean = [clean_text(sample["text"]) for sample in dataset["train"]]
    train_labels = [sample["label"] for sample in dataset["train"]]

    test_texts_clean = [clean_text(sample["text"]) for sample in dataset["test"]]
    test_labels = [sample["label"] for sample in dataset["test"]]

    # Step 3: Split 25K training into 20K train + 5K validation (seeded for reproducibility)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_texts_clean), generator=generator).tolist()

    train_indices = indices[: config.TRAIN_SIZE]
    val_indices = indices[config.TRAIN_SIZE : config.TRAIN_SIZE + config.VAL_SIZE]

    train_texts_split = [train_texts_clean[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]

    val_texts_split = [train_texts_clean[i] for i in val_indices]
    val_labels_split = [train_labels[i] for i in val_indices]

    # Step 4: Build vocabulary from training texts only (20K)
    train_tokenized = [text.split() for text in train_texts_split]
    vocab = Vocabulary(min_freq=config.MIN_FREQ, max_size=config.VOCAB_SIZE)
    vocab.build(train_tokenized)

    # Step 5: Encode all texts using the vocabulary
    train_encoded = [vocab.encode(text.split()) for text in train_texts_split]
    val_encoded = [vocab.encode(text.split()) for text in val_texts_split]
    test_encoded = [vocab.encode(text.split()) for text in test_texts_clean]

    # Step 6: Pad all sequences to MAX_LEN
    train_padded = [pad_sequence(seq, config.MAX_LEN) for seq in train_encoded]
    val_padded = [pad_sequence(seq, config.MAX_LEN) for seq in val_encoded]
    test_padded = [pad_sequence(seq, config.MAX_LEN) for seq in test_encoded]

    # Step 7: Create IMDBDataset instances
    train_dataset = IMDBDataset(train_padded, train_labels_split)
    val_dataset = IMDBDataset(val_padded, val_labels_split)
    test_dataset = IMDBDataset(test_padded, test_labels)

    # Step 8: Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, vocab
