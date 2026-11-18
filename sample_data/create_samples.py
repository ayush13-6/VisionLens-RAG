import os
from pathlib import Path
import cv2
import numpy as np

import config


def generate_invoice_sample(output_path: Path):
    """Generates a synthetic receipt/invoice document with text and table layout."""
    # Create white canvas (800x1000)
    img = np.full((1000, 800, 3), 255, dtype=np.uint8)

    # Header
    cv2.putText(img, "INVOICE & RECEIPT", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (20, 20, 20), 2)
    cv2.putText(img, "Doc ID: INV-2026-9042", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.putText(img, "Date: November 18, 2026", (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Dividing line
    cv2.line(img, (50, 140), (750, 140), (200, 200, 200), 2)

    # Customer info
    cv2.putText(img, "Billed To: CloudTech Solutions Inc.", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 1)
    cv2.putText(img, "Payment Terms: Net 30 Days", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    # Table Header Box
    cv2.rectangle(img, (50, 250), (750, 290), (240, 240, 240), -1)
    cv2.rectangle(img, (50, 250), (750, 290), (180, 180, 180), 1)
    cv2.putText(img, "ITEM DESCRIPTION", (60, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, "QTY", (450, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, "UNIT PRICE", (530, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.putText(img, "TOTAL", (670, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    # Table Rows
    items = [
        ("NVIDIA H100 GPU Cloud Compute (Hours)", "120", "$4.50", "$540.00"),
        ("Vector Database Managed Cluster", "1", "$250.00", "$250.00"),
        ("OpenCV Vision Inference Pipeline", "1", "$180.00", "$180.00"),
        ("Ollama Local LLM Endpoint Setup", "1", "$150.00", "$150.00"),
        ("Enterprise Support Tier", "1", "$80.00", "$80.00"),
    ]

    y_offset = 330
    for desc, qty, price, total in items:
        cv2.putText(img, desc, (60, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
        cv2.putText(img, qty, (460, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
        cv2.putText(img, price, (540, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
        cv2.putText(img, total, (670, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
        cv2.line(img, (50, y_offset + 15), (750, y_offset + 15), (240, 240, 240), 1)
        y_offset += 45

    # Summary box
    cv2.rectangle(img, (480, 580), (750, 680), (245, 248, 255), -1)
    cv2.rectangle(img, (480, 580), (750, 680), (200, 210, 230), 1)
    cv2.putText(img, "Subtotal: $1,200.00", (500, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(img, "Tax (8.5%): $102.00", (500, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(img, "TOTAL DUE: $1,302.00", (500, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 20, 20), 2)

    # Footer note
    cv2.putText(img, "Authorized Signature: A. Sharma", (50, 750), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.8, (20, 20, 100), 1)
    cv2.putText(img, "Thank you for your business! Direct queries to billing@cloudtech.ai", (50, 820), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    cv2.imwrite(str(output_path), img)


def generate_architecture_sample(output_path: Path):
    """Generates a technical architecture presentation slide with modules and boxes."""
    img = np.full((700, 1000, 3), 250, dtype=np.uint8)

    # Title Banner
    cv2.rectangle(img, (0, 0), (1000, 90), (35, 30, 45), -1)
    cv2.putText(img, "VisionLens-RAG: Multimodal Agent Architecture", (40, 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # Subtitle
    cv2.putText(img, "End-to-End Pipeline: OpenCV Vision Preprocessing -> Vector DB -> Ollama LLM", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    # Box 1: Visual Ingestion
    cv2.rectangle(img, (60, 180), (320, 320), (255, 255, 255), -1)
    cv2.rectangle(img, (60, 180), (320, 320), (0, 120, 215), 2)
    cv2.putText(img, "1. OpenCV Vision Layer", (80, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 180), 2)
    cv2.putText(img, "- 4-Point Perspective Warp", (80, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "- Bilateral Denoising Filter", (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "- Otsu & Adaptive Binarization", (80, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Arrow 1
    cv2.arrowedLine(img, (325, 250), (385, 250), (0, 120, 215), 3, tipLength=0.3)

    # Box 2: RAG Vector Store
    cv2.rectangle(img, (390, 180), (650, 320), (255, 255, 255), -1)
    cv2.rectangle(img, (390, 180), (650, 320), (16, 124, 65), 2)
    cv2.putText(img, "2. Multimodal Vector Store", (405, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 124, 65), 2)
    cv2.putText(img, "- Spatial Layout Chunking", (405, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "- Sublinear TF-IDF + Cosine Sim", (405, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "- Hybrid Re-ranking Retrieval", (405, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Arrow 2
    cv2.arrowedLine(img, (655, 250), (715, 250), (16, 124, 65), 3, tipLength=0.3)

    # Box 3: Ollama Agent
    cv2.rectangle(img, (720, 180), (950, 320), (255, 255, 255), -1)
    cv2.rectangle(img, (720, 180), (950, 320), (180, 50, 120), 2)
    cv2.putText(img, "3. Ollama LLM Agent", (735, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 50, 120), 2)
    cv2.putText(img, "- Local Llama3 / Mistral", (735, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "- Grounded Context Prompting", (735, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
    cv2.putText(img, "- Offline Deterministic Mode", (735, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)

    # Specs box at bottom
    cv2.rectangle(img, (60, 380), (950, 640), (245, 245, 245), -1)
    cv2.rectangle(img, (60, 380), (950, 640), (200, 200, 200), 1)
    cv2.putText(img, "SYSTEM PERFORMANCE BENCHMARKS & SPECS", (80, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    specs = [
        "Processing Latency: OpenCV deskew & binarization in 12ms per 1080p frame",
        "Retrieval Accuracy: 96.4% Top-1 Precision using hybrid spatial re-ranking",
        "Memory Footprint: < 85MB total RAM overhead for vision + vector indexing",
        "Privacy & Compliance: 100% on-premise execution with zero third-party cloud data egress",
        "Deployment Compatibility: Python 3.10 - 3.13 on Windows, Linux, and macOS"
    ]

    for idx, spec in enumerate(specs):
        cv2.putText(img, f">> {spec}", (80, 465 + idx * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    cv2.imwrite(str(output_path), img)


def create_all_samples():
    """Builds sample dataset."""
    sample_dir = config.SAMPLE_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)

    invoice_path = sample_dir / "sample_invoice.png"
    arch_path = sample_dir / "sample_architecture.png"

    generate_invoice_sample(invoice_path)
    generate_architecture_sample(arch_path)

    print(f"Sample images created successfully at:")
    print(f"  - {invoice_path}")
    print(f"  - {arch_path}")


if __name__ == "__main__":
    create_all_samples()
