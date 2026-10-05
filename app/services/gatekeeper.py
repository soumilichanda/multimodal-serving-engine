"""
app/services/gatekeeper.py
Tier-1 Semantic Gatekeeper: Raw byte sanitization and ImageNet Synset OOD Filter.
"""

import base64
import io
from typing import Tuple
from PIL import Image
import numpy as np


class SemanticOODGatekeeper:
    """
    Tier-1 Defense Layer:
    1. Validates magic bytes (JPEG/PNG) to avoid decoder exploits.
    2. Runs ImageNet synset verification: Domestic pets lie in:
       synsets in [151, 268] (canines) U [281, 285] (felines).
    """

    PET_SYNSET_RANGES = [
        (151, 268),  # Domestic dogs
        (281, 285),  # Domestic cats
    ]

    def __init__(self, confidence_threshold: float = 0.08):
        self.confidence_threshold = confidence_threshold

    def validate_magic_bytes(self, b64_str: str) -> bytes:
        """Enforces valid JPEG/PNG headers before decoding."""
        if not (b64_str.startswith("/9j/") or b64_str.startswith("iVBORw0KGgo")):
            raise ValueError("Corrupted image header: Missing valid JFIF/PNG signature.")

        try:
            raw_bytes = base64.b64decode(b64_str)
        except Exception as e:
            raise ValueError(f"Base64 decoding failed: {str(e)}")

        return raw_bytes

    def is_in_distribution(self, top_class_id: int, top_prob: float) -> bool:
        """
        Deterministic range check: Returns True if the detected synset index
        falls into domestic dog or cat classes with probability >= threshold.
        """
        if top_prob < self.confidence_threshold:
            return False

        for low, high in self.PET_SYNSET_RANGES:
            if low <= top_class_id <= high:
                return True
        return False

    def preprocess_image(
        self, raw_bytes: bytes, target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Converts raw bytes to an NCHW normalized float32 tensor."""
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        image = image.resize(target_size)
        arr = np.array(image, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # HWC -> CHW -> NCHW
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)