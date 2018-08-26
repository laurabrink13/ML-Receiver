import multiprocessing as mp
import numpy as np


def _data_generator(radio_transmitter, transform_func, omega, snr_dB, batch_size,
                    seed=None, num_cpus=4):
    """A generic generator for providing radio data in parallel. 

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
            signals = pool.map(radio_transmitter.emit_signal,
                               [(seed + i if seed else None) for i in range(batch_size)])
            inputs, labels = transform_func(signals, omega, snr_dB, seed)
            yield inputs, labels

    except Exception as e:
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