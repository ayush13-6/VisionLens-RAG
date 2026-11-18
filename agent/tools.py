import os
from pathlib import Path
from typing import Dict, List, Optional
import cv2

import config
from rag.document_chunker import DocumentChunker
from rag.retriever import HybridRetriever
from rag.vector_store import VectorStore
from vision.preprocessor import DocumentPreprocessor
from vision.region_detector import RegionDetector
from vision.visual_features import VisualFeatureExtractor


class AgentTools:
    """
    Modular toolkit exposed to the Vision-RAG Agent.
    Bridges OpenCV image processing pipelines and RAG vector store operations.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.preprocessor = DocumentPreprocessor()
        self.region_detector = RegionDetector()
        self.feature_extractor = VisualFeatureExtractor()
        self.chunker = DocumentChunker()
        self.retriever = HybridRetriever(self.vector_store)

    def preprocess_image(self, image_path: str, base_name: str = "doc") -> Dict:
        """
        Tool: Normalizes and enhances document images using OpenCV deskewing,
        bilateral filtering, and adaptive binarization.
        """
        return self.preprocessor.process_pipeline(image_path, base_name=base_name)

    def detect_layout_regions(self, image_path: str) -> Dict:
        """
        Tool: Segments document visual layout into headers, text blocks, and tables/diagrams.
        """
        img = self.preprocessor.load_image(image_path)
        regions, annotated = self.region_detector.detect_regions(img)
        annotated_path = self.region_detector.save_annotated(
            annotated, f"annotated_{Path(image_path).name}"
        )
        return {
            "total_regions": len(regions),
            "regions": regions,
            "annotated_image_path": annotated_path
        }

    def inspect_quality(self, image_path: str) -> Dict:
        """
        Tool: Measures visual sharpness, contrast, edge density, and OCR readiness.
        """
        img = self.preprocessor.load_image(image_path)
        stats = self.feature_extractor.extract_image_stats(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = self.preprocessor.measure_sharpness(gray)
        stats["sharpness_score"] = round(sharpness, 2)
        stats["is_sharp"] = bool(sharpness > 80.0)
        return stats

    def ingest_document(
        self,
        image_path: str,
        title: str,
        text_content: Optional[str] = None
    ) -> Dict:
        """
        Tool: Full ingestion pipeline.
        1. Preprocesses image via OpenCV (deskew + enhance)
        2. Detects spatial layout regions
        3. Extracts visual features & OCR text
        4. Generates multimodal chunks with spatial bounding box metadata
        5. Indexes into vector database
        """
        base_name = Path(image_path).stem
        prep_result = self.preprocessor.process_pipeline(image_path, base_name=base_name)
        rectified_path = prep_result["rectified_image_path"]

        # Run layout analysis on the rectified image
        rectified_img = self.preprocessor.load_image(rectified_path)
        regions, annotated = self.region_detector.detect_regions(rectified_img)
        annotated_path = self.region_detector.save_annotated(
            annotated, f"annotated_{base_name}.png"
        )

        # Extract stats
        stats = self.feature_extractor.extract_image_stats(rectified_img)

        # Extract text: user-provided text or OCR
        extracted_text = text_content or self.feature_extractor.extract_text(rectified_img)
        if not extracted_text.strip():
            # Fallback structured descriptor if OCR binary is not present
            extracted_text = (
                f"Document: {title}. "
                f"Layout contains {len(regions)} detected visual sections. "
                f"Image resolution {stats.get('resolution')} with {stats.get('quality_rating')} quality. "
                f"Sections identified: {', '.join([r['type'] for r in regions[:4]])}."
            )

        # Generate multimodal chunks
        doc_id = f"doc_{base_name}_{abs(hash(title)) % 10000}"
        chunks = self.chunker.create_multimodal_chunks(
            doc_id=doc_id,
            doc_title=title,
            text_content=extracted_text,
            source_image=rectified_path,
            visual_regions=regions,
            image_stats=stats
        )

        # Add to vector store
        self.vector_store.add_chunks(chunks)

        return {
            "status": "success",
            "doc_id": doc_id,
            "title": title,
            "chunks_created": len(chunks),
            "rectified_image": rectified_path,
            "annotated_image": annotated_path,
            "detected_regions_count": len(regions),
            "quality_rating": stats.get("quality_rating")
        }

    def search_kb(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        """
        Tool: Queries vector store with hybrid re-ranking.
        """
        return self.retriever.retrieve(query=query, top_k=top_k)

    def capture_webcam(self, filename: str = "webcam_capture.png") -> Dict:
        """
        Tool: Captures a real-time frame from attached webcam.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return {"status": "error", "message": "No accessible webcam hardware detected."}

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"status": "error", "message": "Failed to read frame from webcam."}

        save_path = self.preprocessor.save_image(frame, filename)
        return {
            "status": "success",
            "captured_path": save_path,
            "dimensions": f"{frame.shape[1]}x{frame.shape[0]}"
        }
