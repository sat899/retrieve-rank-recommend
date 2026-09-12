"""
Tests for linalg.py
"""

import numpy as np
from utils.linalg import compute_vector_magnitude, normalize_vector_length

def test_compute_vector_magnitude():
    vector = np.array([3.0, 4.0])
    expected_magnitude = 5.0
    assert compute_vector_magnitude(vector) == expected_magnitude

def test_normalize_vector_length():
    vector = np.array([3.0, 4.0])
    expected_length = 1.0
    normalized_vector = normalize_vector_length(vector)
    assert np.isclose(compute_vector_magnitude(normalized_vector), expected_length)