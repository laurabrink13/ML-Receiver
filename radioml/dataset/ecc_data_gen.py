import commpy as cp
import numpy as np
from radioml.core.channel_simulator import additive_white_gaussian_noise
from radioml.utils import data_generator, encode_complex_to_real

def ecc_data_generator(radio_transmitter, snr_db, batch_size, seed=None, num_cpus=4):
    """
    Returns: `Iterator` object that generates (inputs, outputs) as
        Inputs: equalized_packet
        Outputs: message_bits
    """
    def _ecc_data_func(dataset, snr_db):
        np.random.seed(seed)
        # Unpack  radio data
        packets, modulated_packets = zip(*dataset)
        preamble_len = radio_transmitter.modulated_preamble_len
        packets = np.array(packets)

        # Process inputs
        x = np.array(modulated_packets)[:, preamble_len:]
        noisy_packets = additive_white_gaussian_noise(x, snr_db)
        noisy_packets = encode_complex_to_real(noisy_packets[:, preamble_len:])

        # Process labels:
        vis = encode_complex_to_real(x[:, preamble_len:])
        message_bits = np.expand_dims(packets[:, preamble_len:], -1)
        return [noisy_packets, vis], message_bits
        
    kwargs = {'snr_db': snr_db}
    return data_generator(radio_transmitter, _ecc_data_func, 
                         batch_size, seed, num_cpus, **kwargs)