from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import commpy as cp

from radioml.core.channel_simulator import carrier_frequency_offset
from radioml.dataset.utils import _encode_complex_to_real, _data_generator


def cfo_net_data_generator(radio_transmitter, omega, snr_dB, batch_size,
                           seed=None, num_cpus=4):
    """
    Returns: `Iterator` object that generates (inputs, outputs) as
        Inputs: [preamble, cfo_preamble]
        Outputs: cfo_corrected_preamble
    """

    def _cfo_data_func(dataset, omega, snr_dB, seed=None):
        np.random.seed(seed)

        # Unpack  radio data
        preambles, _, modulated_packets = zip(*dataset)

        batch_size = len(modulated_packets)
        preamble_len = radio_transmitter.get_modulated_preamble_len()
        preambles = np.array(preambles)

        modulated_packets = np.array(modulated_packets)

        # Add AWGN noise
        noisy_packets = cp.channels.awgn(modulated_packets.flatten(), snr_dB)
        noisy_packets = noisy_packets.reshape((batch_size, -1))

        # Simulate CFO
        w_batch = np.random.uniform(-omega, omega, size=(batch_size, 1))
        rotated = carrier_frequency_offset(noisy_packets, w_batch)

        # Process Inputs
        preambles_conv = np.array(modulated_packets)[:, :preamble_len]

        # Process labels
        cfo_corrected_preamble = noisy_packets[:, :preamble_len]
        cfo_corrected_preamble = _encode_complex_to_real(cfo_corrected_preamble)

        return [preambles, preambles_conv], cfo_corrected_preamble

    return _data_generator(radio_transmitter, _cfo_data_func,
                           omega, snr_dB, batch_size, seed, num_cpus)
