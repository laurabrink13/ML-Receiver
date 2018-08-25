import os
import numpy as np
import tensorflow as tf
from commpy.modulation import QAMModem
from commpy.channelcoding import Trellis


def build_trellis_structure(memory=np.array([2]),
                            g_matrix=np.array([[0o7, 0o5]])):
    """Our current assumption about how input is encoded. All signals are encoded
    using Convolution Code at rate 1/2.

    Arguments:
    ----------
        memory : 1D ndarray of ints
            Number of memory elements per input of the convolutional encoder.

        g_matrix : 2D ndarray of ints (octal representation)
            Generator matrix G(D) of the convolutional encoder. Each element of
            G(D) represents a polynomial.

    Returns:
    --------
        trellis - a Trellis structure
    """
    return Trellis(memory=memory, g_matrix=g_matrix)


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


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    """Write summaries with training and evaluation in on plot.


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


def visualize_signals(ax, x, y, groundtruths=None, title=None, min_val=-2, max_val=2):
    ax.scatter(x, y, c=groundtruths, s=300)
    ax.set_xlabel('I-component')
    ax.set_ylabel('Q-component')
    ax.set_title(title)
    ax.axhline()
    ax.axvline()
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)