"""
Functions for calculating similarity / distance between vectors
"""

import numpy as np
from utils.linalg import compute_vector_magnitude

def dot_product(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Calculates the dot product of two 1D vectors
    Vectors must be of the same shape i.e. (n,)
    """
    if vector_a.shape != vector_b.shape:
        raise ValueError(f"vectors must have the same shape: vector a shape is: {vector_a.shape}, vector b shape is: {vector_b.shape}")
    else:
        return float(np.sum(vector_a * vector_b))

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 1D Vectors
    """
    numerator = dot_product(vector_a, vector_b)

    vector_a_length = compute_vector_magnitude(vector_a)
    vector_b_length = compute_vector_magnitude(vector_b)

    denominator = vector_a_length * vector_b_length
    if denominator == 0.0:
        return 0.0

    return float(numerator/denominator)