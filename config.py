from pathlib import Path

MODEL_NAME = "Rostlab/ProstT5"
AGGREGATED_DB_PATH = "data/aggregated/aggregated.duckdb"
TASK_NAME = "solubility"

SPLIT_SEED = 42
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

MAX_LENGTH = 1024 * 2

TOKENIZED_DATA_DIR = Path("data/tokenized")
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
