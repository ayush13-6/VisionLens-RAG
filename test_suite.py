import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from agent.vision_rag_agent import VisionRAGAgent
from rag.document_chunker import DocumentChunker
from rag.vector_store import NumpyTfidfVectorizer, VectorStore
from sample_data.create_samples import create_all_samples
from vision.preprocessor import DocumentPreprocessor
from vision.region_detector import RegionDetector
from vision.visual_features import VisualFeatureExtractor


class TestVisionLensRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        create_all_samples()
        cls.sample_invoice = config.SAMPLE_DIR / "sample_invoice.png"
        cls.sample_arch = config.SAMPLE_DIR / "sample_architecture.png"

    def test_01_opencv_preprocessor(self):
        preprocessor = DocumentPreprocessor()
        result = preprocessor.process_pipeline(str(self.sample_invoice), base_name="test_inv")

        self.assertIn("rectified_image_path", result)
        self.assertIn("binarized_image_path", result)
        self.assertTrue(os.path.exists(result["rectified_image_path"]))
        self.assertTrue(os.path.exists(result["binarized_image_path"]))
        self.assertGreater(result["final_sharpness"], 0.0)

    def test_02_opencv_region_detector(self):
        preprocessor = DocumentPreprocessor()
        img = preprocessor.load_image(str(self.sample_invoice))
        detector = RegionDetector()
        regions, annotated = detector.detect_regions(img)

        self.assertGreater(len(regions), 0)
        first = regions[0]
        self.assertIn("bbox", first)
        self.assertIn("normalized_bbox", first)
        self.assertIn("type", first)
        self.assertEqual(annotated.shape[:2], img.shape[:2])

    def test_03_visual_feature_extractor(self):
        extractor = VisualFeatureExtractor()
        preprocessor = DocumentPreprocessor()
        img = preprocessor.load_image(str(self.sample_arch))
        stats = extractor.extract_image_stats(img)

        self.assertIn("resolution", stats)
        self.assertIn("brightness", stats)
        self.assertIn("contrast", stats)
        self.assertIn("edge_density", stats)
        self.assertIn(stats["quality_rating"], ["High", "Medium", "Low"])

    def test_04_numpy_vector_store(self):
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        chunks = chunker.create_multimodal_chunks(
            doc_id="test_doc_1",
            doc_title="Cloud Server Spec",
            text_content="NVIDIA H100 GPU compute instances are available at $4.50 per hour.",
            source_image=str(self.sample_invoice),
            visual_regions=[{"region_id": 1, "type": "header", "bbox": {"x": 10, "y": 10, "w": 100, "h": 20}}]
        )

        test_store_path = config.STORAGE_DIR / "test_vector_store.json"
        store = VectorStore(storage_path=test_store_path)
        store.clear()
        store.add_chunks(chunks)

        results = store.search("NVIDIA H100 cost", top_k=1)
        self.assertEqual(len(results), 1)
        chunk, score = results[0]
        self.assertIn("NVIDIA H100", chunk.text)
        self.assertGreater(score, 0.0)

        # Clean up
        store.clear()

    def test_05_vision_rag_agent_end_to_end(self):
        agent = VisionRAGAgent()
        res_ingest = agent.scan_and_ingest(
            image_path=str(self.sample_invoice),
            title="Invoice Unit Test",
            text_content="TOTAL DUE: $1,302.00 for CloudTech Solutions Inc."
        )
        self.assertEqual(res_ingest["status"], "success")

        # Query
        qa_result = agent.ask("What is the total due amount?")
        self.assertIn("answer", qa_result)
        self.assertGreater(len(qa_result["retrieved_sources"]), 0)


if __name__ == "__main__":
    unittest.main()
