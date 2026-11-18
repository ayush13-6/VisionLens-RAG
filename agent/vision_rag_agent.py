import logging
from typing import Any, Dict, List, Optional

import config
from rag.retriever import HybridRetriever
from rag.vector_store import VectorStore
from .ollama_client import OllamaRAGClient
from .tools import AgentTools

logger = logging.getLogger("VisionRAGAgent")


class VisionRAGAgent:
    """
    Autonomous Vision-RAG Agent.
    Combines OpenCV computer vision preprocessing, layout segmentation,
    vector retrieval, and Ollama LLM synthesis into a unified reasoning agent.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        ollama_client: Optional[OllamaRAGClient] = None
    ):
        self.vector_store = vector_store or VectorStore()
        self.ollama = ollama_client or OllamaRAGClient()
        self.tools = AgentTools(vector_store=self.vector_store)
        self.retriever = HybridRetriever(self.vector_store)

    def scan_and_ingest(
        self,
        image_path: str,
        title: str,
        text_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Agent Workflow:
        1. Validates and preprocesses image via OpenCV.
        2. Detects layout regions.
        3. Indexes multimodal chunks into RAG vector store.
        """
        logger.info(f"Agent ingesting visual document: {title} from {image_path}")
        result = self.tools.ingest_document(
            image_path=image_path,
            title=title,
            text_content=text_content
        )
        return result

    def inspect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Agent Workflow: Runs visual inspection audit (sharpness, edges, layout).
        """
        quality = self.tools.inspect_quality(image_path)
        layout = self.tools.detect_layout_regions(image_path)
        return {
            "quality": quality,
            "layout": layout
        }

    def ask(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Agent Workflow:
        1. Formulates retrieval query.
        2. Retrieves relevant multimodal chunks with hybrid re-ranking.
        3. Builds context-grounded prompt.
        4. Synthesizes answer using Ollama LLM (or deterministic fallback).
        """
        reasoning_trace = []

        # Step 1: Retrieval
        reasoning_trace.append(f"Step 1: Searching vector store for visual document chunks matching '{query}'")
        retrieved_items = self.retriever.retrieve(query=query, top_k=top_k)

        reasoning_trace.append(f"Step 2: Retrieved {len(retrieved_items)} relevant visual context chunks")

        # Step 2: Context Construction
        context_str = self.retriever.format_context_for_llm(retrieved_items)

        # Step 3: LLM Prompt Formulation
        system_prompt = (
            "You are VisionLens, an expert Multimodal AI Document Agent. "
            "You analyze visual documents, scanned receipts, system diagrams, and technical notes. "
            "Answer the user query accurately and concisely based ONLY on the provided retrieved context. "
            "Always cite the source document, visual region (e.g. header, table, text block), and image name."
        )

        user_prompt = f"{context_str}\n\nUser Query: {query}\n\nPlease provide a clear, factual answer with source citations:"

        reasoning_trace.append(f"Step 3: Querying Ollama LLM ({self.ollama.model})")
        answer = self.ollama.generate(prompt=user_prompt, system_prompt=system_prompt)

        return {
            "query": query,
            "answer": answer,
            "retrieved_sources": retrieved_items,
            "reasoning_trace": reasoning_trace,
            "ollama_connected": self.ollama.is_connected,
            "model_used": self.ollama.model if self.ollama.is_connected else "Offline RAG Extractor"
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns health and status of all components."""
        store_stats = self.vector_store.get_stats()
        return {
            "agent_name": "VisionLens-RAG Agent",
            "opencv_version": "cv2 integrated",
            "ollama_connected": self.ollama.check_connection(),
            "ollama_base_url": self.ollama.base_url,
            "ollama_model": self.ollama.model,
            "available_models": self.ollama.available_models,
            "vector_store": store_stats
        }
