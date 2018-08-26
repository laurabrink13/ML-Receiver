from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import commpy as cp

from radioml.core.channel_simulator import additive_white_gaussian_noise
from radioml.core.channel_simulator import channel_interference
from radioml.utils import encode_complex_to_real, data_generator


def equalization_data_generator(radio_transmitter, num_channels, snr_db, 
                                batch_size, seed=None, num_cpus=4):
    """
    Returns: `Iterator` object that generates (inputs, outputs) as
        Inputs: [preamble, cfo_corected_preamble, cfo_corrected_data],
        Outputs: equalized_packet
    """

    def _process_data_func_for_equalization_net(dataset, num_channels, snr_db):
        np.random.seed(seed)

        # Unpack  radio data
        _, modulated_packets = zip(*dataset)

        modulated_packets = np.array(modulated_packets)
        preamble_len =  radio_transmitter.modulated_preamble_len

        # Add AWGN noise
        noisy_packets = additive_white_gaussian_noise(modulated_packets, snr_db)

        # Simulate multi-tap channel interference
        convolved_packets, channels = channel_interference(noisy_packets, num_channels)

        # Process Inputs
        preambles = encode_complex_to_real(modulated_packets[:, :preamble_len])
        preamble_conv = encode_complex_to_real(convolved_packets[:, :preamble_len])
        data_conv = encode_complex_to_real(convolved_packets[:, preamble_len:])

        # Process Label
        equalized_packet = encode_complex_to_real(noisy_packets[:, preamble_len:])
        modulated_packets = encode_complex_to_real(modulated_packets)

        return [preambles, preamble_conv, data_conv], \
               [equalized_packet, modulated_packets]  # np.array(channels)]

    kwargs = {'num_channels': num_channels, 'snr_db': snr_db}
    return data_generator(radio_transmitter, _process_data_func_for_equalization_net,
                          batch_size, seed, num_cpus, **kwargs)
