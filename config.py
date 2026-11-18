import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = BASE_DIR / "sample_data"
STORAGE_DIR = BASE_DIR / "storage"
PROCESSED_DIR = STORAGE_DIR / "processed_images"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm:135m")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# Vector Store / RAG Configuration
VECTOR_STORE_PATH = STORAGE_DIR / "vector_store.json"
DEFAULT_TOP_K = 3
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50

# OpenCV Preprocessing Defaults
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150
GAUSSIAN_BLUR_KERNEL = (5, 5)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Web UI Configuration
WEB_HOST = "127.0.0.1"
WEB_PORT = 8000
