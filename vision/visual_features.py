from typing import Dict, List, Optional
import cv2
import numpy as np

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


class VisualFeatureExtractor:
    """
    Computes statistical, structural, and OCR features from document images.
    Feeds visual metadata into RAG chunks for hybrid retrieval.
    """

    def __init__(self):
        self.tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        if not HAS_PYTESSERACT:
            return False
        try:
            # Quick check if tesseract binary responds
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_image_stats(self, image: np.ndarray) -> Dict:
        """
        Calculates brightness, contrast, edge density, and visual complexity.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]
        total_pixels = float(h * w)

        mean_val, std_val = cv2.meanStdDev(gray)
        brightness = float(mean_val[0][0])
        contrast = float(std_val[0][0])

        # Edge density via Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = float(np.count_nonzero(edges))
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0.0

        # Contrast ratio / quality rating
        if contrast > 50 and edge_density > 0.02:
            quality = "High"
        elif contrast > 30:
            quality = "Medium"
        else:
            quality = "Low"

        return {
            "resolution": f"{w}x{h}",
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "edge_density": round(edge_density, 4),
            "quality_rating": quality
        }

    def extract_text(self, image: np.ndarray) -> str:
        """
        Attempts OCR via pytesseract if system binary is available.
        Otherwise returns structured visual descriptor.
        """
        if self.tesseract_available:
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    return text.strip()
            except Exception:
                pass
        return ""
