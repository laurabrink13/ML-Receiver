"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing as mp
import numpy as np
import commpy as cp

from radioml.core.radio_transmitter import RadioTransmitter
from radioml.core.channel_simulator import channel_interference, carrier_frequency_offset


class RadioDataGenerator(object):
    """Radio Data Generator is responsible for simulating radio data being sent over an
    Additive White Gaussian Noise (AWGN) Channel. In addition, there are also Inter-symbol Interference and Carrier
    Frequency Offset that affects the signals.

                                        output = H.input + noise

    Arguments:
    ----------


    """
    def __init__(self, data_len, preamble_len, channels_len, modulation_scheme='qpsk'):
        self.radio_transmitter = RadioTransmitter(data_len,
                                                  preamble_len,
                                                  channels_len,
                                                  modulation_scheme)


    def ecc_data_generator(self, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """
        Returns: `Iterator` object that generates (inputs, outputs) as
            Inputs: equalized_packet
            Outputs: message_bits
        """

        def _process_data_for_demod_n_ecc_net(dataset, omega, snr_dB, seed=None):
            np.random.seed(seed)

            # Unpack  radio data
            preambles, message_bits, modulated_packets = zip(*dataset)
            preamble_len = self.radio_transmitter.get_modulated_preamble_len()

            # Process inputs
            x = np.array(modulated_packets)[:, preamble_len:]
            noisy = cp.channels.awgn(x.flatten(), snr_dB).reshape((len(x), -1))
            equalized_packet = self._encode_complex_to_real(noisy)

            # Process labels:
            message_bits = np.expand_dims(message_bits, -1)

            vis = self._encode_complex_to_real(x)
            return [equalized_packet, vis], message_bits

        return self._data_genenerator(_process_data_for_demod_n_ecc_net,
                                      omega, snr_dB, batch_size, seed, num_cpus)

    def end2end_data_generator(self, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """
        Returns: `Iterator` object that generates (inputs, outputs) as

            Inputs: [preamble, corrupted_packet],
            Outputs: original_message_bits
        """

        def _process_data_end2end_net(dataset, omega, snr_dB, seed=None):
            np.random.seed(seed)

            # Unpack  radio data
            original_packets, modulated_packets = zip(*dataset)
            batch_size = len(modulated_packets)
            preamble_len = self.get_modulated_preamble_len()

            # Simulate multi-tap channel interference
            convolved_packets, channels = self._channel_interefence(modulated_packets)

            # Add AWGN noise
            noisy_packets = cp.channels.awgn(convolved_packets.flatten(), snr_dB)
            noisy_packets = noisy_packets.reshape((batch_size, -1))

            # Simulate CFO
            w_batch = np.random.uniform(-omega, omega, size=(batch_size, 1))
            rotated = self._carrier_frequency_offset(noisy_packets, w_batch)

            # Process inputs
            preambles = np.array(modulated_packets)[:, :preamble_len]
            preambles = self._encode_complex_to_real(preambles)
            corrupted_packets = self._encode_complex_to_real(rotated)

            # Process labels
            orignal_message_bits = np.expand_dims(
                np.array(original_packets)[:, preamble_len:], -1)

            return [preambles, corrupted_packets], \
                   [orignal_message_bits, w_batch, channels]

        return self._data_genenerator(_process_data_end2end_net,
                                      omega, snr_dB, batch_size, seed, num_cpus)


def _data_generator(transform_func, omega, snr_dB, batch_size, seed=None, num_cpus=4):
    """A generic generator returns an `Iterator` object that
    generates (inputs, labels) until it raises a `StopIteration` exception,

    Arguments:
    ----------
        transform_func: callable function that generate (inputs, labels)
        omega (float): angular frequency (in radian, e.g. 1/50, 1/100)
        snr_dB(float): signal-to-noise ratio in Decibel
        batch_size(int): number of samples per training/eval step
        seed  (int):
        num_cpus (int): number of cpu cores for generating data in parallel.

    Returns:
    --------
        `Iterator` object that yields (inputs, labels)
    """
    pool = mp.Pool(num_cpus)
    try:
        while True:
            signals = pool.map(self.emit_signal, [(seed + i if seed else None)
                                                  for i in range(batch_size)])
            inputs, labels = transform_func(signals, omega, snr_dB, seed)
            yield inputs, labels
    except Exception as e:
        print(e)
        raise e
    finally:
        pool.close()


def _encode_complex_to_real(inputs):
    """TF does not support complex numbers for training.
    Therefore, we need to encode complex inputs into 2D array.

    Arguments:
    ----------
        inputs: complex ndarray [batch, data_len]

    Return:
    -------
        encoded_inputs: float ndarray  [batch, data_len, 2]
    """

    if isinstance(inputs[0], complex):
        return np.stack([np.real(inputs),
                         np.imag(inputs)], -1)
    else:
        return np.array(inputs)