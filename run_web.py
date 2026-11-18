import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_data.create_samples import create_all_samples
from ui.web_app import start_server

if __name__ == "__main__":
    create_all_samples()
    start_server()
