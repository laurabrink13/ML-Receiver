import numpy as np

from radioml.core.channel_simulator import channel_interference
from radioml.core.channel_simulator import carrier_frequency_offset
from radioml.core.channel_simulator import additive_white_gaussian_noise
from radioml.utils import data_generator, encode_complex_to_real

def end2end_data_generator(radio_transmitter, num_channels, omega, snr_db, 
                           batch_size, seed=None, num_cpus=4):
    """
    Returns: `Iterator` object that generates (inputs, outputs) as

        Inputs: [preamble, corrupted_packet],
        Outputs: original_message_bits
    """
    def _end2end_data_func(dataset, num_channels, omega, snr_dB):
        np.random.seed(seed)

        # Unpack  radio data
        original_packets, modulated_packets = zip(*dataset)
        batch_size = len(modulated_packets)
        preamble_len = radio_transmitter.modulated_preamble_len

        # Simulate multi-tap channel interference
        convolved_packets, channels = channel_interference(modulated_packets, num_channels)

        noisy_packets = additive_white_gaussian_noise(convolved_packets, snr_dB)

        # Simulate CFO
        w_batch = np.random.uniform(-omega, omega, size=(batch_size, 1))
        rotated = carrier_frequency_offset(noisy_packets, w_batch)

        # Process inputs
        preambles = np.array(modulated_packets)[:, :preamble_len]
        preambles = encode_complex_to_real(preambles)
        corrupted_packets = encode_complex_to_real(rotated)

        # Process labels
        orignal_message_bits = np.expand_dims(
            np.array(original_packets)[:, preamble_len:], -1)

        return [preambles, corrupted_packets], \
                [orignal_message_bits, w_batch, channels]

    kwargs = {'num_channels': num_channels, 'omega': omega, 'snr_db': snr_db}

    return data_generator(radio_transmitter, _end2end_data_func, batch_size, 
                          seed, num_cpus, **kwargs )
