"""
Tests for similarity.py
"""

import numpy as np
from utils.similarity import dot_product

def test_dot_product():
    vector_a = np.array([3.0, 5.0])
    vector_b = np.array([6.0, 2.0])
    expected_result = 28.0
    assert np.isclose(dot_product(vector_a, vector_b), expected_result)