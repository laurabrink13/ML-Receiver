from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.signal as sig


def channel_interference(inputs, channel_length):
    """Simulate multi-tap channel interference on a batch of data stream.

    Arguments:
    ----------
        inputs: complex nd-array: [batch, data_length]

    Returns:
    --------
        convolved_inputs: complex nd-array
    """
    # @TODO : assert input shape validation
    x = np.random.uniform(-1, 1, (len(inputs), channel_length))
    channels = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]

    # @TODO : vector-ize this op
    a = [sig.convolve(x, y, mode='same') for x, y in zip(inputs, channels)]
    return np.array(a), channels


def carrier_frequency_offset(inputs, omegas):
    """Simulate Carrier frequency offset (CFO) @ some omega on a batch of data stream.

    Argument:
    ---------
        inputs: complex nd-array: [batch, data_length]
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
    batch_size, data_len = np.shape(inputs)
    time_steps_matrix = np.tile(np.arange(data_len), (batch_size, 1))
    rotated_packets = inputs * np.exp(1j * omegas * time_steps_matrix)
    return rotated_packets
