import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from agent.vision_rag_agent import VisionRAGAgent
from sample_data.create_samples import create_all_samples


def print_banner():
    banner = """
================================================================================
           VISIONLENS-RAG : MULTIMODAL DOCUMENT INTELLIGENCE AGENT              
            [ OpenCV Vision Engine + Vector Store + Ollama Local LLM ]         
================================================================================
    """
    print(banner)


def run_demo():
    print_banner()

    # Step 1: Initialize Agent and System Status
    print("[1/5] Initializing VisionLens-RAG Agent...")
    agent = VisionRAGAgent()
    status = agent.get_status()
    print(f"  -> Agent Status: Ready")
    print(f"  -> Ollama Host: {status['ollama_base_url']}")
    print(f"  -> Ollama Live: {'CONNECTED' if status['ollama_connected'] else 'OFFLINE (Engaging Grounded Fallback)'}")
    print(f"  -> Target Model: {status['ollama_model']}")
    print()

    # Step 2: Ensure Sample Data Exists
    print("[2/5] Preparing sample visual documents with OpenCV...")
    create_all_samples()
    invoice_img = config.SAMPLE_DIR / "sample_invoice.png"
    arch_img = config.SAMPLE_DIR / "sample_architecture.png"
    print()

    # Step 3: Visual Inspection & Layout Analysis
    print("[3/5] Running OpenCV visual inspection & layout analysis on Sample Invoice...")
    inspection = agent.inspect_image(str(invoice_img))
    q = inspection["quality"]
    l = inspection["layout"]
    print(f"  -> Resolution: {q.get('resolution')}")
    print(f"  -> Sharpness Score: {q.get('sharpness_score')} (Clear: {q.get('is_sharp')})")
    print(f"  -> Contrast Score: {q.get('contrast')}")
    print(f"  -> Detected Layout Regions: {l.get('total_regions')}")
    for r in l.get("regions", [])[:4]:
        print(f"     * Region [{r['region_id']}]: {r['type']} at {r['bbox']}")
    print(f"  -> Annotated Image Saved: {l.get('annotated_image_path')}")
    print()

    # Step 4: Multimodal Ingestion into RAG Vector Store
    print("[4/5] Ingesting documents into RAG Vector Store...")
    # Document 1: Invoice
    invoice_text = (
        "INVOICE & RECEIPT. Doc ID: INV-2026-9042. Date: November 18, 2026. "
        "Billed To: CloudTech Solutions Inc. Payment Terms: Net 30 Days. "
        "Line Items: "
        "1. NVIDIA H100 GPU Cloud Compute (120 Hours at $4.50/hr): $540.00. "
        "2. Vector Database Managed Cluster (1 unit): $250.00. "
        "3. OpenCV Vision Inference Pipeline (1 unit): $180.00. "
        "4. Ollama Local LLM Endpoint Setup (1 unit): $150.00. "
        "5. Enterprise Support Tier: $80.00. "
        "Financial Summary: Subtotal is $1,200.00. Tax (8.5%) is $102.00. TOTAL DUE is $1,302.00. "
        "Authorized Signature: A. Sharma. Contact: billing@cloudtech.ai"
    )
    res_inv = agent.scan_and_ingest(
        image_path=str(invoice_img),
        title="Enterprise GPU Infrastructure Invoice",
        text_content=invoice_text
    )
    print(f"  -> Ingested Invoice: {res_inv['title']} ({res_inv['chunks_created']} chunks)")

    # Document 2: Architecture Spec
    arch_text = (
        "VisionLens-RAG Multimodal Agent Architecture Specification. "
        "Pipeline components: "
        "1. OpenCV Vision Layer: 4-Point Perspective Warp, Bilateral Denoising Filter, Otsu & Adaptive Binarization. "
        "2. Multimodal Vector Store: Spatial Layout Chunking, Sublinear TF-IDF + Cosine Similarity, Hybrid Re-ranking. "
        "3. Ollama LLM Agent: Local Llama3 / Mistral, Grounded Context Prompting, Offline Deterministic Fallback. "
        "Performance Benchmarks: "
        "Processing Latency: OpenCV deskew and binarization takes 12ms per 1080p frame. "
        "Retrieval Accuracy: 96.4% Top-1 Precision using hybrid spatial re-ranking. "
        "Memory Footprint: Under 85MB total RAM overhead for vision and vector indexing. "
        "Privacy: 100% on-premise execution with zero cloud data egress."
    )
    res_arch = agent.scan_and_ingest(
        image_path=str(arch_img),
        title="VisionLens-RAG System Architecture & Benchmarks",
        text_content=arch_text
    )
    print(f"  -> Ingested Spec: {res_arch['title']} ({res_arch['chunks_created']} chunks)")
    print(f"  -> Total Vector Store Chunks: {agent.vector_store.get_stats()['total_chunks']}")
    print()

    # Step 5: Multi-hop RAG Queries
    print("[5/5] Executing Multimodal RAG Queries...")
    queries = [
        "What is the total amount due on the invoice and who is it billed to?",
        "What are the processing latency and memory footprint benchmarks of the system?",
        "How much does the NVIDIA H100 GPU compute cost per hour according to the bill?"
    ]

    for idx, q in enumerate(queries, start=1):
        print(f"\n--- [Query {idx}]: \"{q}\" ---")
        start_time = time.time()
        result = agent.ask(q)
        elapsed = (time.time() - start_time) * 1000

        print(f"Model Engine : {result['model_used']}")
        print(f"Latency      : {elapsed:.1f}ms")
        print(f"Retrieved    : {len(result['retrieved_sources'])} sources")
        for s in result["retrieved_sources"][:2]:
            print(f"  - [{s['doc_title']}] (Region: {s['region_type']}, Score: {s['relevance_score']:.2f})")
        print(f"\nAnswer:\n{result['answer']}")
        print("-" * 70)

    print("\n[SUCCESS] VisionLens-RAG end-to-end demonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
