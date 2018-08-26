import os
import multiprocessing as mp

import numpy as np
import tensorflow as tf
from commpy.modulation import QAMModem
from commpy.channelcoding import Trellis


def generate_synthetic_packet(preamble_len, data_len, seed):
    """Generate one synthetic packet.
    Arguments:
    ----------
        preamble_len: int - size of the preamble in a packet
        data_len: int - size of the data in a packet
    Returns:
    --------
        packet : ndarray size [data_len + preamble_len, 1]
    """
    np.random.seed(seed)
    preamble = np.random.randint(0, 2, preamble_len)
    message_bits = np.random.randint(0, 2, data_len)
    packet = np.concatenate([preamble, message_bits])
    return packet


def build_modulator(modulation_scheme):
    """Construct a modulator.
    Arguments:
    ----------
        modulation_scheme: a string - name of a modulation scheme.
    Returns:
    --------
        QAMModem - represents  modulator @ particular modulation scheme.
    """
    # @TODO: add more modulation schemes

    if str.lower(modulation_scheme) == 'qpsk':
        return QAMModem(m=4)
    elif str.lower(modulation_scheme) == 'qam16':
        return QAMModem(m=16)
    elif str.lower(modulation_scheme) == 'qam64':
        return QAMModem(m=64)
    elif str.lower(modulation_scheme) == 'qam128':
        return QAMModem(m=128)
    else:
        raise ValueError('Modulation scheme {} is not supported'.format(modulation_scheme))


def build_trellis_structure(num_shift=1, num_output=2, constraint_len=3):
    """Construct a Trellis Structure. 
    
    Our current assumption for this project 
    is all signals are encoded using Convolution Code at rate 1/2.

    Arguments:
    ----------
        num_shift : int -  number of bits shifted into the encoder at one time.
        num_output: int -  number of encoder output bits corresponding to the 
                           `num_shift` information bits.
        constraint_len: int - encoder memory
        
    Returns:
    --------
        trellis - a Trellis structure
    """
    
    data_rate = num_shift / num_output
    memory = np.array([constraint_len - 1])
    g_matrix = _build_g_matrix(data_rate, constraint_len)
    return Trellis(memory=memory, g_matrix=g_matrix)


def data_generator(radio_transmitter, transform_func, batch_size, seed=None, num_cpus=4, **kwargs):
    """A generic generator for providing radio data in parallel. 

    Arguments:
    ----------
        transform_func: callable function that generate (inputs, labels)
        batch_size(int): number of samples per training/eval step
        seed  (int):
        num_cpus (int): number of cpu cores for generating data in parallel.
        **kwargs: parameters that would pass into transform_func
    Returns:
    --------
        `Iterator` object that yields (inputs, labels)
    """
    pool = mp.Pool(num_cpus)
    try:
        while True:
            signals = pool.map(radio_transmitter.emit_signal,
                               [(seed + i if seed else None) for i in range(batch_size)])
            inputs, labels = transform_func(signals, **kwargs)
            yield inputs, labels
    except Exception as e:
        raise e
    finally:
        pool.close()


def encode_complex_to_real(inputs):
    """TF does not support complex numbers for training.
    Therefore, we need to encode complex inputs into 2D array.

    Arguments:
    ----------
        inputs: complex ndarray [batch, data_len]

    Return:
    -------
        encoded_inputs: float ndarray  [batch, data_len, 2]
    """

    if np.iscomplexobj(inputs):
        return np.stack([np.real(inputs),
                         np.imag(inputs)], -1)
    else:
        return np.array(inputs)


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    """Write summaries with training and evaluation in on plot.
    Source:
    https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
    """
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def _build_g_matrix(data_rate, constraint_len):
    # @TODO: generalize how to construct G matrix based on
    # data_rate and constraint_len ??
    if data_rate == 1/2:
        if constraint_len == 3:
            g_matrix = np.array([[0o7, 0o5]])
        elif constraint_len == 4:
            g_matrix = np.array([[0o17, 0o13]])
        else:
            raise ValueError('Not support current trellis structure.')
    elif data_rate == 1/3:
        if constraint_len == 4:
            g_matrix = np.array([[0o13, 0o15, 0o17]])
    else:
        raise ValueError('Not support current trellis structure.')

    return g_matrix