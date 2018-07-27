import numpy as np
import commpy as cp
from radioml.models import Baseline
from radioml.dataset import generate_signal_over_awgn

def test_baseline_on_low_snr():
    np.random.seed(2018)

    qpsk = cp.modulation.QAMModem(4)
    trellis = cp.channelcoding.Trellis(memory=np.array([2]), 
                                       g_matrix=np.array([[0o7, 0o5]]))
    baseline = Baseline(trellis, modulation_scheme=qpsk)
    message_bits, _, corrupted_complex = generate_signal_over_awgn(qpsk, trellis, 
                                                                   snr_dB=6.0)
    data_estimate = baseline(corrupted_complex)
    data_estimate = data_estimate[:100]

    hamming_dist = cp.utilities.hamming_dist(message_bits, data_estimate)
    assert hamming_dist == 10.0


def test_baseline_on_high_snr():
    np.random.seed(2018)

    qpsk = cp.modulation.QAMModem(4)
    trellis = cp.channelcoding.Trellis(memory=np.array([2]), 
                                       g_matrix=np.array([[0o7, 0o5]]))
    baseline = Baseline(trellis, modulation_scheme=qpsk)
    message_bits, _, corrupted_complex = generate_signal_over_awgn(qpsk, trellis, 
                                                                   snr_dB=15.0)
    data_estimate = baseline(corrupted_complex)
    data_estimate = data_estimate[:100]

    hamming_dist = cp.utilities.hamming_dist(message_bits, data_estimate)
    assert hamming_dist == 0.0
