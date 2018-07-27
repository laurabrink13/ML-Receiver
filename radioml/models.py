
"""Contains implementations of Radio Receivers."""
import numpy as np
from commpy.modulation import QAMModem
from commpy.channelcoding import viterbi_decode

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, BatchNormalization, Activation
from tensorflow.keras.layers import RepeatVector, Bidirectional, TimeDistributed

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
        * Demodulator (QPSK, QAM16, PSK, etc.)
        * Decoder: Viterbi
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

    def equalize(self, inputs):
        raise NotImplementedError
        
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

        def cfo_network(preamble, preamble_conv, scope='CFOCorrectionNet'):
            """
            Arguments:
                preamble :     tf.Tensor float32 -  [batch, preamble_length, 2]
                preamble_conv: tf.Tensor float32 -  [batch, preamble_length, 2]
                
            Return:
                cfo_estimate: tf.Tensor float32 - [batch_size, 1]
            """
            with tf.name_scope(scope):
                inputs = tf.keras.layers.concatenate([preamble, preamble_conv], axis=1)
                inputs = tf.keras.layers.Flatten(name='Flatten')(inputs)
                x = tf.keras.layers.Dense(100, 'selu', name=scope+"_dense_1")(inputs)
                x = tf.keras.layers.Dense(100, 'selu', name=scope+"_dense_2")(x)
                x = tf.keras.layers.Dense(100, 'selu', name=scope+"_dense_3")(x)
            cfo_est = tf.keras.layers.Dense(1, 'linear',name='CFOEstimate')(x)
            return cfo_est


        def cfo_correction(kwargs):
            """Given an CFO estimate w, rotate packets in opposite w as

                packets = packets * e^(-j*w*range(len(packets)))
            
            Arguments:
                omega_estimate: tf.Tensor float32 - [batch, 1]
                packets:        tf.Tensor float32 - [batch, (preamble_len + data_len), 2] 
                
            Return:
                rotated_packets: tf.Tensor float32 - [batch, (preamble_len + data_len), 2] 
            """ 
            # Because of Lambda Layer, we need to pass arguments as Kwargs
            omega_estimate, packets = kwargs[0], kwargs[1]
            with tf.name_scope('CFOCorrection'):
                with tf.name_scope('rotation_matrix'):
                    # preamble_len + data_len
                    packet_len      = tf.cast(tf.shape(packets)[1], tf.float32)
                    rotation_matrix = tf.exp(tf.complex(0.0, - 1.0 * omega_estimate * tf.range(packet_len)))
                with tf.name_scope('cfo_correction'):
                    rotated_packets = tf.complex(packets[..., 0], packets[...,1]) * rotation_matrix

                # Encode complex packets into 2D array
                rotated_packets = tf.stack([tf.real(rotated_packets), 
                                            tf.imag(rotated_packets)], 
                                        axis=-1, name='cfo_corrected')
            return rotated_packets


        def equalization_network(cfo_corrected_packets, preamble):
            with tf.name_scope('EqualizationNet'):
                inputs = tf.keras.layers.concatenate([preamble, cfo_corrected_packets], axis=1)
                x = Bidirectional(LSTM(20, return_sequences=True))(inputs)
                x = Bidirectional(LSTM(20, return_sequences=False))(x)
                x = Dense(400, activation='relu')(x)
                
                x = Dense(400, activation='linear')(x)
                equalized_packets = Reshape((200, 2))(x)
            return equalized_packets

        def demod_and_ecc_network(equalized_packets):
            num_hidden_layers = 2
            hidden_units=400 # 400
            with tf.name_scope('DemodAndDecodeNet'):
                x = equalized_packets
                for _ in range(num_hidden_layers):
                    x = Bidirectional(GRU(hidden_units, return_sequences=True))(x)
                    x = BatchNormalization()(x)
            data_estimates = TimeDistributed(Dense(1, activation='sigmoid'), name='DataEstimate')(x)
            return data_estimates