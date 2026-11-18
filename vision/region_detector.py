from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import config


class RegionDetector:
    """
    OpenCV document layout analysis and visual region segmentation.
    Uses morphological kernels to detect text lines, blocks, tables, and diagrams,
    attaching spatial coordinates and visual attributes for multimodal RAG retrieval.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or config.PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_regions(
        self,
        image_input,
        min_area: int = 500,
        max_regions: int = 15
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detects structural layout regions (text blocks, headers, tables/diagrams).
        Returns list of region descriptors and an annotated visualization image.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not load image: {image_input}")
        else:
            img = image_input.copy()

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert binarized text: characters become white on black background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological dilation to merge nearby characters and words into cohesive text blocks
        # We use a horizontal rectangular kernel to connect words along lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(thresh, kernel_h, iterations=2)

        # Vertical kernel to merge lines in the same paragraph
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        merged_blocks = cv2.dilate(dilated, kernel_v, iterations=2)

        # Find external contours of visual blocks
        contours, _ = cv2.findContours(merged_blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = img.copy()
        regions = []

        # Filter and sort contours top-to-bottom, left-to-right (natural reading order)
        valid_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Exclude full-page border contours
            if bw > 0.95 * w and bh > 0.95 * h:
                continue
            valid_boxes.append((x, y, bw, bh, area))

        # Sort primarily by vertical Y position (with tolerance) then X position
        valid_boxes.sort(key=lambda b: (b[1] // 30, b[0]))
        valid_boxes = valid_boxes[:max_regions]

        for idx, (x, y, bw, bh, area) in enumerate(valid_boxes):
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0
            norm_y = y / h
            norm_h = bh / h

            # Classify region based on visual and geometric heuristics
            if norm_y < 0.20 and aspect_ratio > 3.0:
                region_type = "header"
                color = (0, 165, 255)  # Orange
            elif bh > 0.25 * h and bw > 0.40 * w:
                region_type = "table_or_diagram"
                color = (255, 0, 128)  # Purple
            elif aspect_ratio > 4.5 and bh < 40:
                region_type = "single_line"
                color = (0, 255, 0)  # Green
            else:
                region_type = "text_block"
                color = (255, 100, 0)  # Blue-ish

            region_crop = img[y : y + bh, x : x + bw]

            # Draw visual bounding box and label
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            label = f"[{idx+1}] {region_type}"
            cv2.putText(
                annotated,
                label,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

            regions.append({
                "region_id": idx + 1,
                "type": region_type,
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "normalized_bbox": {
                    "ymin": round(y / h, 4),
                    "xmin": round(x / w, 4),
                    "ymax": round((y + bh) / h, 4),
                    "xmax": round((x + bw) / w, 4)
                },
                "area": int(area),
                "aspect_ratio": round(aspect_ratio, 2)
            })

        return regions, annotated

    def save_annotated(self, annotated_img: np.ndarray, filename: str) -> str:
        """Saves annotated region image."""
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), annotated_img)
        return str(save_path)
