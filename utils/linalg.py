"""
Functions implementing basic linear algebra operations
"""

import numpy as np

def compute_vector_magnitude(vector: np.ndarray) -> float:
    """
    Calculates the Euclidean length (L2 norm) of a 1D vector.
    This is basically pythagoras' theorem i.e. the square root of all dimensions squared and summed.
    Equivalent to np.linalg.norm in numpy.
    """
    return float(np.sqrt(np.sum(vector ** 2)))

def normalize_vector_length(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a 1D vector by its Euclidean length (L2 norm) to unit length (1)
    """
    vector_magnitude = compute_vector_magnitude(vector)
    
    if vector_magnitude == 0.0:
        return vector.astype(float)
    else:
        return vector / vector_magnitude