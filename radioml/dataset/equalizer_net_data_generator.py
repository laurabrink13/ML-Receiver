from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import commpy as cp

from radioml.core.channel_simulator import channel_interference
from radioml.dataset.utils import _encode_complex_to_real, _data_generator


def equalization_data_generator(radio_transmitter, omega, snr_dB, batch_size, seed=None, num_cpus=4):
    """
    Returns: `Iterator` object that generates (inputs, outputs) as
        Inputs: [preamble, cfo_corected_preamble, cfo_corrected_data],
        Outputs: equalized_packet
    """

    def _process_data_func_for_equalization_net(dataset, omega, snr_dB, seed=None):
        np.random.seed(seed)

        # Unpack  radio data
        preambles, _, modulated_packets = zip(*dataset)

        preambles = np.array(preambles)
        modulated_packets = np.array(modulated_packets)
        batch_size = len(modulated_packets)
        preamble_len =  radio_transmitter.get_modulated_preamble_len()

        # Add AWGN noise
        noisy_packets = cp.channels.awgn(modulated_packets.flatten(), snr_dB)
        noisy_packets = noisy_packets.reshape((batch_size, -1))

        # Simulate multi-tap channel interference
        convolved_packets, channels = channel_interference(noisy_packets)

        # Process Inputs

        preamble_conv = convolved_packets[:, :preamble_len]
        preamble_conv = _encode_complex_to_real(preamble_conv)

        data_conv = convolved_packets[:, preamble_len:]
        data_conv = _encode_complex_to_real(data_conv)

        # Process Label
        x = noisy_packets[:, preamble_len:]
        equalized_packet = _encode_complex_to_real(x)

        return [preambles, preamble_conv, data_conv], \
               [equalized_packet, np.array(channels)]

    return _data_genenerator(radio_transmitter, _process_data_func_for_equalization_net,
                             omega, snr_dB, batch_size, seed, num_cpus)
