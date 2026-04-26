from pathlib import Path

### Model Settings
# Hugging Face model identifier for the frozen ProstT5 encoder backbone.
MODEL_NAME = "Rostlab/ProstT5"
# DuckDB database path containing the aggregated multitask samples and task metadata.
AGGREGATED_DB_PATH = "data/aggregated/aggregated.duckdb"
# Default task name used by legacy single-task cache paths.
TASK_NAME = "solubility"

### Data Split Settings
# Random seed used when shuffling sequences into train, validation, and test splits.
SPLIT_SEED = 42
# Fraction of sequences assigned to the training split.
TRAIN_FRACTION = 0.8
# Fraction of sequences assigned to the validation split.
VAL_FRACTION = 0.1
# Fraction of sequences assigned to the test split.
TEST_FRACTION = 0.1

### Tokenization Settings
# Maximum token length allowed during sequence tokenization before truncation.
MAX_LENGTH = 1024 * 2

### Cache Paths
# Directory where tokenized dataset artifacts are written and loaded from.
TOKENIZED_DATA_DIR = Path("data/tokenized")
# Default single-task tokenized cache path derived from the selected task name.
TOKENIZED_DATA_PATH = TOKENIZED_DATA_DIR / f"{TASK_NAME}_prostt5_tokens.pt"

### Model Settings & Hyperparameters
# Number of sequences per optimization step during training.
BATCH_SIZE = 32
# Learning rate used by AdamW for adapter, pooling, and head parameters.
LR = 1e-4
# Maximum number of full passes over the training split.
EPOCHS = 10
# Bottleneck width for the trainable adapter inserted on top of ProstT5 embeddings.
ADAPTER_DIM = 64
# Dropout probability used in the adapter and attention pooling layers.
DROPOUT = 0.1
# Hidden dimension for the attention pooling MLP.
ATTN_POOL_HIDDEN = 256
# L2-style regularization strength for AdamW.
WEIGHT_DECAY = 1e-2
# Fraction of total training steps used for learning-rate warmup.
WARMUP_RATIO = 0.05
# Number of non-improving validation epochs allowed before early stopping.
PATIENCE = 3
# Random seed used by the length-aware weighted batch sampler.
BATCH_SAMPLER_SEED = 42
# Path to the pre-tokenized multitask cache consumed by training.
TRAIN_CACHE_PATH = TOKENIZED_DATA_DIR / "multitask_prostt5_tokens.pt"
# Maximum padded tokens allowed in evaluation batches to avoid very long-sequence OOMs.
EVAL_MAX_TOKENS_PER_BATCH = 32768
