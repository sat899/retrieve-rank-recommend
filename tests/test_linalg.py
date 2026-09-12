"""
Tests for linalg.py
"""

import numpy as np
from utils.linalg import compute_vector_magnitude

def test_compute_vector_magnitude():
    vector = np.array([3.0, 4.0])
    expected_magnitude = 5.0
    assert compute_vector_magnitude(vector) == expected_magnitude