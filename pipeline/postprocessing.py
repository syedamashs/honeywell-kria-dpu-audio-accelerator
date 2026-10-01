"""
pipeline/postprocessing.py
--------------------------
Post-processing: softmax, argmax, label decoding.

Softmax is NOT in the DPU model (DPUCZDX8G does not support it).
It always runs on the CPU — in Config B/C via VART, the xmodel
outputs raw logits which are passed here.
"""

from __future__ import annotations

import numpy as np

from pipeline.utils import IDX2LABEL, NUM_CLASSES


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def decode(logits: np.ndarray) -> tuple[str, float, int]:
    """
    Args:
        logits: (num_classes,) raw logits from model
    Returns:
        (label, confidence, class_index)
    """
    probs = softmax(logits)
    idx   = int(np.argmax(probs))
    return IDX2LABEL[idx], float(probs[idx]), idx


def decode_batch(logits: np.ndarray) -> list[tuple[str, float, int]]:
    """
    Args:
        logits: (B, num_classes)
    Returns:
        list of (label, confidence, class_index)
    """
    return [decode(logits[i]) for i in range(logits.shape[0])]
