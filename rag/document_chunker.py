from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import uuid

import config


@dataclass
class MultimodalChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    source_image: str
    visual_metadata: Dict = field(default_factory=dict)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MultimodalChunk":
        return cls(**data)


class DocumentChunker:
    """
    Intelligent multimodal chunking engine.
    Splits document contents while binding OpenCV spatial and visual metadata.
    """

    def __init__(self, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Splits raw text into sliding window segments with overlap."""
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            if chunk_str.strip():
                chunks.append(chunk_str)
            if i + self.chunk_size >= len(words):
                break
            i += max(1, self.chunk_size - self.overlap)
        return chunks

    def create_multimodal_chunks(
        self,
        doc_id: str,
        doc_title: str,
        text_content: str,
        source_image: str,
        visual_regions: Optional[List[Dict]] = None,
        image_stats: Optional[Dict] = None
    ) -> List[MultimodalChunk]:
        """
        Creates multimodal chunks mapped to OpenCV detected regions and visual stats.
        """
        raw_chunks = self.chunk_text(text_content)
        if not raw_chunks and text_content.strip():
            raw_chunks = [text_content.strip()]

        multimodal_chunks = []
        regions = visual_regions or []
        num_regions = len(regions)

        for idx, text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_chunk_{idx + 1}_{uuid.uuid4().hex[:6]}"

            # Associate chunk with corresponding visual region if available
            region_info = regions[idx % num_regions] if num_regions > 0 else {}
            meta = {
                "chunk_index": idx + 1,
                "region_id": region_info.get("region_id", 1),
                "region_type": region_info.get("type", "text_block"),
                "bbox": region_info.get("bbox", {}),
                "normalized_bbox": region_info.get("normalized_bbox", {}),
                "aspect_ratio": region_info.get("aspect_ratio", 1.0),
                "visual_stats": image_stats or {}
            }

            chunk = MultimodalChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_title=doc_title,
                text=text,
                source_image=source_image,
                visual_metadata=meta
            )
            multimodal_chunks.append(chunk)

        return multimodal_chunks
