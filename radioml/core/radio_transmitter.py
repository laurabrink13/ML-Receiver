"""
"""
import warnings
import numpy as np
import commpy as cp
from radioml.utils import build_modulator, build_trellis_structure
from radioml.utils import generate_synthetic_packet

class RadioTransmitter(object):
    """Simulation of a Radio Transmitter.

    Assumptions:
        * Data is encoded using Convolutional Code.

    Arguments:
    ----------
        data_len:
        preamble_len:
        trellis:
        modulation_scheme:
    """
    def __init__(self, data_len, preamble_len, modulation_scheme='qpsk', trellis=None):
        self.data_len = data_len
        self.preamble_len = preamble_len
        if trellis is None:
            # warnings.warn('Trellis is None. Use default option (data_rate = 1/2)')
            trellis = build_trellis_structure()
        self.trellis = trellis
        self.modulator = build_modulator(modulation_scheme)

    def emit_signal(self, seed=None):
        packet = generate_synthetic_packet(self.preamble_len, self.data_len, seed)
        _, modulated_packet = self._simulate_radio_transmitter(packet)
        return packet, modulated_packet
    
    def _simulate_radio_transmitter(self, packet):
        """Simulate how a packet passing through a radio transmitter."""
        encoded_packet = cp.channelcoding.conv_encode(packet, self.trellis)
        encoded_packet = encoded_packet[:-self.trellis.n * int(self.trellis.total_memory)]
        modulated_packet = self.modulator.modulate(encoded_packet)
        return encoded_packet, modulated_packet

    @property
    def modulated_preamble_len(self):
        """Different modulation schemes have different preamble length.
        """
        len_after_encoder = self.preamble_len * self.trellis.n / self.trellis.k
        len_after_modulator = len_after_encoder / self.modulator.num_bits_symbol

        if not (len_after_modulator).is_integer():
            raise ValueError('Modulate Preamble Length should be an integer. '
                             'Got %f instead' % len_after_modulator)

        return int(len_after_modulator)