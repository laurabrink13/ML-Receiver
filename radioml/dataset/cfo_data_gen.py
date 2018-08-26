import numpy as np
from radioml.core.channel_simulator import additive_white_gaussian_noise
from radioml.core.channel_simulator import carrier_frequency_offset
from radioml.utils import data_generator, encode_complex_to_real

def cfo_net_data_generator(radio_transmitter, omega, snr_db, batch_size,
                           seed=None, num_cpus=4):
    """Generate data for training, evaluating CFO Correction Network.

    Arguments:
    ----------
        radio_transmitter:
        omega:
        snr_db:
        batch_size:

    Returns: 
    --------
        An `Iterator` object generates (inputs, outputs):
            Inputs: [preamble, cfo_preamble]
            Outputs: cfo_corrected_preamble
    """
    def _cfo_data_func(dataset, omega, snr_dB):
        # Unpack transmitted radio data
        _, modulated_packets = zip(*dataset)

        batch_size = len(modulated_packets)
        modulated_packets = np.array(modulated_packets)

        # Generate cfo effect
        np.random.seed(seed)
        omegas = np.random.uniform(-omega, omega, size=(batch_size, 1))

        # Simulate channel (AWGN + CFO)
        noisy_packets = additive_white_gaussian_noise(modulated_packets, snr_dB)
        rotated_packets = carrier_frequency_offset(noisy_packets, omegas)

        # Process outputs
        preamble_len = radio_transmitter.modulated_preamble_len
        preambles = encode_complex_to_real(modulated_packets[:, :preamble_len])
        preamble_conv = encode_complex_to_real(rotated_packets[:, :preamble_len])
        cfo_corrected_preambles = encode_complex_to_real(noisy_packets[:,:preamble_len])
        return [preambles, preamble_conv], cfo_corrected_preambles
    
    kwargs =  {'omega': omega, 'snr_dB': snr_db}
    return data_generator(radio_transmitter, _cfo_data_func, batch_size, seed, 
                          num_cpus, **kwargs)
