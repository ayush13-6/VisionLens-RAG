from typing import Dict, List, Optional, Tuple
import config
from .document_chunker import MultimodalChunk
from .vector_store import VectorStore


class HybridRetriever:
    """
    Hybrid retriever that blends semantic text similarity with visual confidence weighting.
    Generates structured grounding context with visual provenance for LLM synthesis.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        visual_boost: bool = True
    ) -> List[Dict]:
        """
        Executes hybrid retrieval and returns structured results with provenance.
        """
        raw_results = self.vector_store.search(query=query, top_k=top_k * 2)
        if not raw_results:
            return []

        scored_results = []
        q_lower = query.lower()
        is_structure_query = any(w in q_lower for w in ["table", "diagram", "header", "total", "amount", "summary"])

        for chunk, sim_score in raw_results:
            final_score = sim_score
            meta = chunk.visual_metadata

            if visual_boost:
                # Boost if query relates to structure and region matches
                region_type = meta.get("region_type", "")
                if is_structure_query and region_type in ["table_or_diagram", "header"]:
                    final_score += 0.15

                # Quality bonus for sharper / clear visual source
                stats = meta.get("visual_stats", {})
                if stats.get("quality_rating") == "High":
                    final_score += 0.05

            scored_results.append({
                "chunk": chunk,
                "similarity_score": round(sim_score, 4),
                "final_score": round(final_score, 4)
            })

        # Sort by final score and slice top_k
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = scored_results[:top_k]

        formatted = []
        for item in top_results:
            c: MultimodalChunk = item["chunk"]
            v_meta = c.visual_metadata
            formatted.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "text": c.text,
                "source_image": c.source_image,
                "region_type": v_meta.get("region_type", "unknown"),
                "bbox": v_meta.get("bbox", {}),
                "relevance_score": item["final_score"]
            })

        return formatted

    def format_context_for_llm(self, retrieved_items: List[Dict]) -> str:
        """
        Converts retrieved evidence into structured prompt context for Ollama.
        """
        if not retrieved_items:
            return "No relevant visual document context found in knowledge base."

        context_lines = ["=== RETRIEVED MULTIMODAL CONTEXT ==="]
        for idx, item in enumerate(retrieved_items, start=1):
            context_lines.append(
                f"\n[Source {idx}]: {item['doc_title']} (Region: {item['region_type']}, Relevance: {item['relevance_score']:.2f})"
            )
            context_lines.append(f"Image: {item['source_image']}")
            context_lines.append(f"Content: \"{item['text']}\"")
        context_lines.append("\n=====================================")
        return "\n".join(context_lines)
