"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import commpy as cp
from radioml.utils import build_modulator, build_trellis_structure


class RadioTransmitter(object):
    """Simulation of a Radio Transmitter.

    Assumptions:
        * Data is encoded using Convolutional Code.

    Arguments:
    ----------
        data_len:
        preamble_len:
        channels_len:
        modulation_scheme:
        data_rate:
    """
    def __init__(self, data_len, preamble_len, channels_len, modulation_scheme='qpsk'):
        self.trellis = build_trellis_structure()
        self.modulator = build_modulator(modulation_scheme)

        self.data_len = data_len
        self.preamble_len = preamble_len
        self.channels_len = channels_len

    def emit_signal(self, seed=None):
        """Simulate how source data going through a radio transmitter"""

        # Generate preamble and message bits
        np.random.seed(seed)
        preamble = np.random.randint(0, 2, self.preamble_len)
        message_bits = np.random.randint(0, 2, self.data_len)
        packet = np.concatenate([preamble, message_bits])

        # Simulate TX
        encoded_packet = cp.channelcoding.conv_encode(packet, self.trellis)
        encoded_packet = encoded_packet[:-2 * int(self.trellis.total_memory)]
        modulated_packet = self.modulator.modulate(encoded_packet)

        # @TODO: eventually, preamble should be the original one, not modulated version ???
        preamble = modulated_packet[:self.get_modulated_preamble_len()]
        return preamble, message_bits, modulated_packet

    def get_modulated_preamble_len(self):
        length = self.preamble_len * 2 / self.modulator.num_bits_symbol
        if not length.is_integer():
            raise ValueError('Modulate Preamble Length should be an integer. Got %f instead' % length)
        return int(length)