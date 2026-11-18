import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

import config
from .document_chunker import MultimodalChunk

# Standard compact English stop words list
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class NumpyTfidfVectorizer:
    """
    Lightweight, high-performance TF-IDF vectorizer implemented with NumPy.
    Completely dependency-free, avoiding external C-extension DLL policy locks.
    """

    def __init__(self, stop_words: Optional[Set[str]] = STOP_WORDS):
        self.stop_words = stop_words or set()
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\$]+\b", text.lower())
        return [w for w in words if w not in self.stop_words and len(w) > 1]

    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        self.num_docs = len(raw_documents)
        if self.num_docs == 0:
            return np.empty((0, 0))

        tokenized_docs = [self._tokenize(doc) for doc in raw_documents]

        # Build vocabulary
        vocab = {}
        df_counter = Counter()
        for doc in tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df_counter[term] += 1
                if term not in vocab:
                    vocab[term] = len(vocab)

        self.vocab = vocab
        vocab_size = len(vocab)
        if vocab_size == 0:
            return np.empty((self.num_docs, 0))

        # Smooth IDF: log((1 + N) / (1 + df)) + 1
        self.idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in vocab.items():
            self.idf[idx] = math.log((1 + self.num_docs) / (1 + df_counter[term])) + 1.0

        # Build matrix
        matrix = np.zeros((self.num_docs, vocab_size), dtype=np.float32)
        for doc_idx, doc in enumerate(tokenized_docs):
            if not doc:
                continue
            tf_counts = Counter(doc)
            for term, count in tf_counts.items():
                if term in vocab:
                    term_idx = vocab[term]
                    # Sublinear TF scaling: 1 + log(count)
                    sublinear_tf = 1.0 + math.log(count)
                    matrix[doc_idx, term_idx] = sublinear_tf * self.idf[term_idx]

            # L2 normalize document vector
            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm

        return matrix

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        num_docs = len(raw_documents)
        vocab_size = len(self.vocab)
        if num_docs == 0 or vocab_size == 0:
            return np.empty((num_docs, vocab_size))

        matrix = np.zeros((num_docs, vocab_size), dtype=np.float32)
        for doc_idx, doc_str in enumerate(raw_documents):
            tokens = self._tokenize(doc_str)
            tf_counts = Counter(tokens)
            for term, count in tf_counts.items():
                if term in self.vocab:
                    term_idx = self.vocab[term]
                    sublinear_tf = 1.0 + math.log(count)
                    matrix[doc_idx, term_idx] = sublinear_tf * self.idf[term_idx]

            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm

        return matrix


class VectorStore:
    """
    Lightweight, persistent Vector Database for Multimodal RAG.
    Employs NumPy TF-IDF vectorization, Cosine Similarity ranking,
    spatial-visual metadata filtering, and atomic JSON persistence.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or config.VECTOR_STORE_PATH
        self.chunks: List[MultimodalChunk] = []
        self.vectorizer: Optional[NumpyTfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.load()

    def add_chunks(self, new_chunks: List[MultimodalChunk]):
        """Adds multimodal chunks and re-fits index."""
        if not new_chunks:
            return

        existing_ids = {c.chunk_id for c in self.chunks}
        for nc in new_chunks:
            if nc.chunk_id not in existing_ids:
                self.chunks.append(nc)
                existing_ids.add(nc.chunk_id)

        self._rebuild_index()
        self.save()

    def _rebuild_index(self):
        """Re-indexes corpus using NumPy TF-IDF."""
        if not self.chunks:
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        corpus = [c.text for c in self.chunks]
        self.vectorizer = NumpyTfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        top_k: int = config.DEFAULT_TOP_K,
        filter_region_type: Optional[str] = None
    ) -> List[Tuple[MultimodalChunk, float]]:
        """
        Retrieves top_k most similar chunks for query string using Cosine Similarity.
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        if query_vec.shape[1] == 0 or np.linalg.norm(query_vec) == 0:
            # Query has no known vocabulary tokens, return top documents
            return [(c, 0.0) for c in self.chunks[:top_k]]

        # Compute cosine similarity: dot product of normalized vectors
        similarities = np.dot(self.tfidf_matrix, query_vec.T).flatten()

        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            chunk = self.chunks[idx]

            # Optional metadata filter
            if filter_region_type:
                region_type = chunk.visual_metadata.get("region_type", "")
                if region_type.lower() != filter_region_type.lower():
                    continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    def save(self):
        """Persists knowledge base to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in self.chunks]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self):
        """Loads knowledge base from disk if available."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = [MultimodalChunk.from_dict(item) for item in data]
                self._rebuild_index()
            except Exception:
                self.chunks = []
                self.vectorizer = None
                self.tfidf_matrix = None

    def clear(self):
        """Clears all indexed chunks."""
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        if self.storage_path.exists():
            try:
                os.remove(self.storage_path)
            except OSError:
                pass

    def get_stats(self) -> Dict:
        """Returns statistics about indexed vector store."""
        docs = set(c.doc_id for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(docs),
            "storage_path": str(self.storage_path)
        }
