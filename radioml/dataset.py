import multiprocessing as mp
import numpy as np
import commpy as cp


def generate_signal_over_awgn(modem, trellis, data_length=100, snr_dB=15.0):
    """Simulate data over AWGN Channel."""
    
    message_bits   = np.random.randint(0, 2, data_length)
    
    encoded_bits   = cp.channelcoding.conv_encode(message_bits, trellis)
    
    modulated_complex = modem.modulate(encoded_bits)
    
    # Channel Conv.
    # data_conv_complex
    
    # CFO
    # cfo_complex
    #     
    corrupted_complex = cp.channels.awgn(modulated_complex, snr_dB, rate=1/2)

    return message_bits, modulated_complex, corrupted_complex


def channels_convole()