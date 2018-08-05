import numpy as np
import commpy as cp

def get_ber_bler(estimated_bits, original_bits):
    """Compute Bit Error Rate and Block Error Rate."""
    n_sequences = len(original_bits)
    hamming_distances = []
    for i in range(n_sequences):       
        dist = cp.utilities.hamming_dist(
            original_bits[i].astype(int),
            estimated_bits[i].astype(int))
        hamming_distances.append(dist)
    ber = np.sum(hamming_distances) / np.product(np.shape(original_bits))
    bler = np.count_nonzero(hamming_distances) / len(original_bits)
    return ber, bler