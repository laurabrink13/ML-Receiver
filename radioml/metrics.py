import numpy as np
import commpy as cp


def get_ber_bler(estimated_bits, original_bits):
    """Compute Bit Error Rate and Block Error Rate.

    Arguments:
    ----------
        estimated_bits: int - size [batch, data_len]
        original_bits:  int - size [batch, data_len]

    Returns:
        ber: float - Bit Error Rate
        bler: float - Block Error Rate
    """
    hamming_distances = []
    for i in range(len(original_bits)):
        distance = cp.utilities.hamming_dist(original_bits[i].astype(int), 
                                             estimated_bits[i].astype(int))
        hamming_distances.append(distance)
    ber = np.sum(hamming_distances) / np.product(np.shape(original_bits))
    bler = np.count_nonzero(hamming_distances) / len(original_bits)
    return ber, bler
