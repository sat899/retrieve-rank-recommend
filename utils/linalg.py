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