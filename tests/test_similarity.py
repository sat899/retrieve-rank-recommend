"""
Tests for similarity.py
"""

import numpy as np
from utils.similarity import dot_product, cosine_similarity, jaccard_similarity

def test_dot_product():
    vector_a = np.array([3.0, 5.0])
    vector_b = np.array([6.0, 2.0])
    expected_result = 28.0
    assert np.isclose(dot_product(vector_a, vector_b), expected_result)

def test_cosine_similarity():
    """
    First we calculate the dot product which is 2 + 8 + 18  = 28
    Then the norm of vector_a = sqrt(14) = 3.74...
    The norm of vector_b  = sqrt(56) = 7.48..
    So 28 / 3.74.. * 7.48.. = 1.00
    """
    vector_a = np.array([1.0, 2.0, 3.0])
    vector_b = np.array([2.0, 4.0, 6.0]) # b is just a*2
    expected_result = 1.0
    assert np.isclose(cosine_similarity(vector_a, vector_b), expected_result)

def test_jaccard_similarity():
    vector_a = np.array([1.0, 2.0, 3.0])
    vector_b = np.array([2.0, 4.0, 6.0])
    expected_result = 0.2 # 1/ 5
    assert np.isclose(jaccard_similarity(vector_a, vector_b), expected_result)