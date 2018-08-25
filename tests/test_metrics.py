import numpy as np
from radioml.metrics import get_ber_bler

def test_ber_bler():
    original_bits =  np.array([[1, 1 ,0], [1, 0, 1], [0, 0, 0], [1, 1, 0]])
    estimated_bits = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1]])
    bit_error_rate, block_error_rate = get_ber_bler(estimated_bits, original_bits)
    assert bit_error_rate == 0.5
    assert block_error_rate == 0.75