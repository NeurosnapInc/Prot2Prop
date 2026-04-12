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
