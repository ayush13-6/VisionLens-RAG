import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

import config


class DocumentPreprocessor:
    """
    OpenCV-powered visual document enhancement and geometric normalization.
    Performs edge detection, perspective warping (deskewing), adaptive thresholding,
    bilateral filtering, and sharpness evaluation.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: str) -> np.ndarray:
        """Safely loads an image from disk."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Use imdecode to safely handle Windows unicode paths
        with open(image_path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"OpenCV failed to decode image: {image_path}")
        return img

    def save_image(self, img: np.ndarray, filename: str) -> str:
        """Saves processed image to the processed storage directory."""
        save_path = self.output_dir / filename
        ext = os.path.splitext(filename)[1] or ".png"
        is_success, buffer = cv2.imencode(ext, img)
        if not is_success:
            raise IOError(f"Failed to encode image to {save_path}")
        with open(save_path, "wb") as f:
            f.write(buffer)
        return str(save_path)

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders coordinates in: [top-left, top-right, bottom-right, bottom-left]
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has smallest diff
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has largest diff
        return rect

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Applies 4-point perspective transform to extract flattened, top-down view.
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width of new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Compute height of new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Guard against zero dimensions
        max_width = max(max_width, 100)
        max_height = max(max_height, 100)

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Compute perspective transform matrix and warp
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped

    def detect_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Locates the largest quadrilateral contour representing document boundaries.
        """
        orig_h, orig_w = image.shape[:2]
        # Resize for faster edge detection while keeping aspect ratio
        ratio = orig_h / 500.0
        resized = cv2.resize(image, (int(orig_w / ratio), 500))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        edged = cv2.Canny(blurred, config.CANNY_LOW_THRESH, config.CANNY_HIGH_THRESH)

        # Close gaps between edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Quadrilateral with sufficient area
            if len(approx) == 4 and cv2.contourArea(c) > (resized.shape[0] * resized.shape[1] * 0.15):
                doc_contour = approx.reshape(4, 2) * ratio
                break

        return doc_contour

    def measure_sharpness(self, gray: np.ndarray) -> float:
        """
        Measures image sharpness using the variance of the Laplacian.
        Higher value indicates sharper text / edges.
        """
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def enhance_contrast_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def process_pipeline(self, image_path: str, base_name: str = "doc") -> Dict:
        """
        Runs the complete visual preprocessing pipeline:
        1. Load & Evaluate Sharpness
        2. Detect Document Boundaries & Perspective Warp (deskew)
        3. Bilateral Filter Denoising
        4. Adaptive Otsu Binarization
        5. Saves processed assets and returns structured metadata
        """
        orig = self.load_image(image_path)
        h, w = orig.shape[:2]
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        initial_sharpness = self.measure_sharpness(gray)

        # Step 1: Document Boundary Detection & Perspective Warp
        contour = self.detect_document_contour(orig)
        is_warped = False
        if contour is not None:
            processed_color = self.four_point_transform(orig, contour)
            is_warped = True
        else:
            processed_color = orig.copy()

        # Step 2: Grayscale and Bilateral Filtering (Edge-preserving noise reduction)
        warped_gray = cv2.cvtColor(processed_color, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(
            warped_gray,
            config.BILATERAL_D,
            config.BILATERAL_SIGMA_COLOR,
            config.BILATERAL_SIGMA_SPACE
        )

        # Step 3: Contrast Enhancement
        clahe_enhanced = self.enhance_contrast_clahe(denoised)

        # Step 4: Adaptive / Otsu Thresholding for crisp text binarization
        _, otsu_bin = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_bin = cv2.adaptiveThreshold(
            clahe_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )

        final_sharpness = self.measure_sharpness(warped_gray)

        # Save artifacts
        color_path = self.save_image(processed_color, f"{base_name}_rectified.png")
        bin_path = self.save_image(otsu_bin, f"{base_name}_binarized.png")
        adaptive_path = self.save_image(adaptive_bin, f"{base_name}_adaptive.png")

        return {
            "source_path": image_path,
            "rectified_image_path": color_path,
            "binarized_image_path": bin_path,
            "adaptive_image_path": adaptive_path,
            "is_warped": is_warped,
            "initial_dimensions": {"width": w, "height": h},
            "processed_dimensions": {
                "width": processed_color.shape[1],
                "height": processed_color.shape[0]
            },
            "initial_sharpness": round(initial_sharpness, 2),
            "final_sharpness": round(final_sharpness, 2),
            "is_clear": bool(final_sharpness > 80.0)
        }
