import os
import tempfile
from pathlib import Path

_sentiment_dir = Path(__file__).resolve().parent

# Logs go to temp dir (writable on Streamlit Cloud)
DB_LOGS = os.getenv("DB_LOGS", str(Path(tempfile.gettempdir()) / "production_logs.db"))

# Warehouse only needed for training (not used in inference)
DB_WAREHOUSE = os.getenv("DB_WAREHOUSE", str(_sentiment_dir / "corporate_data_warehouse.db"))

# SVM model pkl lives alongside this file in the repo
SVM_MODEL_PATH = os.getenv("SVM_MODEL_PATH", str(_sentiment_dir / "svm_model.pkl"))

# BERT model: if not found locally, transformer_predict.py falls back to HuggingFace Hub
BERT_MODEL_DIR = os.getenv("BERT_MODEL_DIR", str(_sentiment_dir / "bert_model_saved"))
