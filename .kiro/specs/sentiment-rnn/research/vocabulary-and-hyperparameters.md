# Research: Vocabulary Size and Hyperparameters for IMDB Sentiment Analysis

## Dataset Statistics
- IMDB dataset: 50,000 reviews total (25K train, 25K test)
- Median review length: ~174 words ([Google ML Guide](https://developers.google.com/machine-learning/guides/text-classification/step-2))
- Total unique words in raw dataset: ~90,000-100,000+ (before cleaning)

## Common Vocabulary Sizes in Practice

| Source | vocab_size | max_len | Notes |
|--------|-----------|---------|-------|
| Keras official LSTM example | 20,000 | 80 | Minimal, fast training |
| Keras hyperparameter tuning example | 20,000 | 100 | Standard baseline |
| SHAP LSTM example | 20,000 | 80 | "Dataset too small for LSTM advantage" |
| StackOverflow common practice | 10,000 | 200 | Longer sequences, smaller vocab |
| PyTorch tutorials (various) | 25,000 | 200-500 | More common in PyTorch implementations |

Sources: [SHAP docs](https://shap.readthedocs.io/en/stable/example_notebooks/text_examples/sentiment_analysis/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.html), [DataScience StackExchange](https://datascience.stackexchange.com/questions/9823)

## Analysis

### Vocabulary Size
- **10,000**: Aggressive filtering. Covers ~95% of token occurrences but loses rare/domain-specific words.
- **20,000**: Most common choice in literature. Good balance between coverage and embedding matrix size.
- **25,000-30,000**: Slightly better coverage, marginal accuracy improvement (~0.5-1%), larger embedding matrix.
- **50,000+**: Diminishing returns. Many low-frequency words that don't help generalization.

### Sequence Length (max_len)
- **80-100**: Fast training, but truncates many reviews (median is 174 words).
- **200**: Captures most reviews without excessive padding. Common sweet spot.
- **300-500**: Captures nearly all reviews but increases training time significantly.
- Reviews longer than 500 words are rare (~5% of dataset).

### Other Common Hyperparameters
- **Embedding dim**: 100-300 (128 or 256 most common)
- **Hidden size**: 128-256
- **Num layers**: 2 (for stacked RNN)
- **Dropout**: 0.3-0.5
- **Batch size**: 32-64
- **Learning rate**: 0.001 (Adam default)
- **Epochs**: 5-10 (overfitting starts around epoch 5-7 for IMDB)

## Recommendation
For a pedagogical project comparing LSTM vs GRU:
- **vocab_size = 25,000** — good coverage without excessive memory
- **max_len = 256** — captures most reviews, good for learning about padding effects
- **embedding_dim = 128** — reasonable for the vocab size
- **hidden_size = 256** — standard for 2-layer stacked RNN
- **min_freq = 2** — filter words appearing only once (reduces noise)
