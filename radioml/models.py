
"""Contains implementations of Radio Receivers."""
import numpy as np
from commpy.modulation import QAMModem
from commpy.channelcoding import viterbi_decode

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.layers import GRU, Bidirectional, TimeDistributed


class RadioReceiver(object):
    """Abstract Radio Receiver class."""
     
    def __init__(self):
        pass

    def __call__(self, noisy_signals):
        """Estimate noisy signals

        Arguments:
            noisy_signals (complex/float ndarray): 

        Return:
            data_estimate (int ndarray): estimate of original message bits
        """
        raise NotImplementedError


class Baseline(RadioReceiver):
    """
        * Equalizer @TODO: add MMSE
        * Demodulator (QPSK, QAM16, PSK, or Neural-network based model)
        * Decoder (Viterbi, MAP, or Neural-network based model)
    """

    def __init__(self, 
                 trellis, 
                 modulation_scheme='QPSK', 
                 tb_depth=15, 
                 decoding_type='hard'):
        """
        Initialize Baseline Radio Receiver.

        Arguments:
            trellis:
            modulation_scheme:
            tb_depth:
            decoding_type:
        """
        super(Baseline, self).__init__()
        self.modulator = QAMModem(m=4)
        self.trellis = trellis
        self.tb_depth = tb_depth
        self.decoding_type= decoding_type

    def __call__(self, complex_inputs):
        # @TODO: add parallel processing
        decoded_bits  = self.demodulate(complex_inputs)
        estimated_message_bits = self.decode(decoded_bits)
        return estimated_message_bits

    def demodulate(self, inputs):
        return self.modulator.demodulate(inputs, demod_type='hard')

    def decode(self, inputs):
        return viterbi_decode(inputs, self.trellis, self.tb_depth, 
                              self.decoding_type)


class End2End(RadioReceiver):
    """Add Documentation."""

    def __init__(self, data_length, pretrained_model_path=None):
        self.data_length = data_length
        self.model = self._build_model(pretrained_model_path)

    def __call__(self, inputs, batch_size):
        x = self.model.predict(self._preprocess_fn(inputs), batch_size)
        predictions = np.squeeze(x, -1).round()
        return predictions

    def _preprocess_fn(self, complex_inputs):
        # Encode complex inputs to 2D float ndarray.
        # Example: 
        #    [1 + -1j, 0 + 1j] --> [[1, -1], [0, 1]] 8
        x = np.stack((np.array(complex_inputs).real,
                      np.array(complex_inputs).imag),
                      axis=-1)
        return x.reshape((-1, self.data_length, 2))  

    def _build_model(self, pretrained_model_path):

        # Define network parameters (num layers, hidden units, etc.)
        num_hidden_layers = 2
        hidden_units = 400
        dropout = 0.3
        
        # Define network architecture
        inputs  = Input(shape=(None, 2))
        x = inputs
        for _ in range(num_hidden_layers):
            x = Bidirectional(GRU(units=hidden_units,
                                  return_sequences=True, 
                                  recurrent_dropout=dropout))(x)
            x = BatchNormalization()(x)
        outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

        # Convert to a `keras.Model` for training/evaluation.
        model = tf.keras.Model(inputs, outputs)

        # Load pretrained weights
        if pretrained_model_path:
            model.load_weights(pretrained_model_path)
            
        return model