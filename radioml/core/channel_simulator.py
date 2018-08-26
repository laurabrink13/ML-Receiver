import numpy as np
import commpy as cp
import scipy.signal as sig


def additive_white_gaussian_noise(data_stream, snr_db):
    """Add white gaussian nosie to data stream.

    Arguments:
    ----------
        data_stream: complex or float nd-array: [batch, data_length]
        snr_db: float - signal-to-noise ratio

    Returns:
    --------
        noisy_data_stream: complex or float nd-array [batch, data_length]
    """
    batch_size = len(data_stream)
    return cp.channels.awgn(data_stream.flatten(), snr_db).reshape((batch_size, -1))


def channel_interference(data_stream, channel_length):
    """Simulate multi-tap channel interference on a batch of data stream.

    Arguments:
    ----------
        data_stream: complex nd-array: [batch, data_length]

    Returns:
    --------
        convolved_inputs: complex nd-array [batch, data_length]

    """
    # @TODO : assert input shape validation
    x = np.random.uniform(-1, 1, (len(data_stream), channel_length))
    channels = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]

    # @TODO : vector-ize this op
    a = [sig.convolve(x, y, mode='same') for x, y in zip(data_stream, channels)]
    return np.array(a), channels


def carrier_frequency_offset(data_stream, omegas):
    """Simulate Carrier frequency offset (CFO) @ some omega on a batch of data stream.

    Argument:
    ---------
        data_stream: complex nd-array: [batch, data_length]
        omegas: float array [1, batch]

    Returns:
    -------
        rotated_packets: complex nd-array

    Notes:
    ------
        We define a time_step_matrix as the following example:

         If batch_size = 3, data_len = 5
         --> time_steps_matrix =[[0, 1, 2, 3, 4],
                                 [0, 1, 2, 3, 4],
                                 [0, 1, 2, 3, 4]]
    """
    # @TODO : assert input shape validation
    batch_size, data_len = np.shape(data_stream)
    time_steps_matrix = np.tile(np.arange(data_len), (batch_size, 1))
    rotated_packets = data_stream * np.exp(1j * omegas * time_steps_matrix)
    return rotated_packets

