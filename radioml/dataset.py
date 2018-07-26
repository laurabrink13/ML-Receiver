import multiprocessing as mp
import numpy as np
import commpy as cp
import scipy.signal as sig
import math

class RadioData(object):
    """Simulate sending data over multi-tap channels with carrier frequency
    offset (CFO)."""

    def __init__(self, 
                 modem, 
                 trellis, 
                 data_len=100, 
                 preamble_len=40, 
                 channels_len=2, 
                 data_rate=1/2):
        """
        Initalize Radio Data object

        Arguments:
            TODO
        """
        self.modem = modem
        self.trellis = trellis
        self.data_len = data_len
        self.preamble_len = preamble_len
        self.channels_len = channels_len
        self.data_rate = data_rate

    def generate_packet(self, 
                        omega = 1/100,
                        snr_dB=15.0):
        """Simulate data over AWGN Channel."""
        
        # Generate preamble and message bits
        preamble       = np.random.randint(0, 2, self.preamble_len)
        message_bits   = np.random.randint(0, 2, self.data_len)

        # shape: [preamble_len + data_len, 1]
        packet         = np.concatenate([preamble, message_bits])

        # Simulate TX

        # shape: [2*(preamble_len + datal_len) + 4, 1]
        encoded_packet   = cp.channelcoding.conv_encode(packet, self.trellis)

        # shape: [preamble_len + data_len + 2, 1]
        modulated_packet = self.modem.modulate(encoded_packet)

        # Simulate multi-tap channel interference
        channels = np.random.uniform(0, 1, self.channels_len)  # -1 to 1
        channels = channels / channels.sum()  # normalize to sum of one
        convolved_packet = sig.convolve(modulated_packet, channels, mode='same')

        # Simulate Carrier frequency offset (CFO) @ some omega
        w = np.random.uniform(low=-omega, high=omega)  # omega = 1/1000

        # TODO: is it correct?
        cfo = np.exp(1j *  w * np.arange(len(convolved_packet)))
        rotated_packet = convolved_packet * cfo

        # Simulate packet sending over AWGN channel @ some signal-to-noise ratio
        corrupted_packet = cp.channels.awgn(rotated_packet, snr_dB)

        return packet, w, modulated_packet, rotated_packet, corrupted_packet