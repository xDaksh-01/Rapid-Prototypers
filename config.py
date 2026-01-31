
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "massive_transactions.csv")
LABELS_PATH = os.path.join(DATA_DIR, "massive_labels.csv")

# Model Settings
MODEL_PATH = os.path.join(BASE_DIR, "aml_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "aml_scaler.pkl")

# Graph Settings
MAX_GRAPH_NODES = 500  # Limit for visualization to avoid lag
DEFAULT_HOP_DEPTH = 1
