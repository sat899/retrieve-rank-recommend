"""
Functions for calculating similarity / distance between vectors
"""

import numpy as np

def dot_product(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculates the dot product of two 1D vectors
    Vectors must be of the same shape i.e. (n,)
    """
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"vectors must have the same shape: vector a shape is: {vector_a.shape}, vector b shape is: {vector_b.shape}")
    else:
        return float(np.sum(vector_a * vector_b))