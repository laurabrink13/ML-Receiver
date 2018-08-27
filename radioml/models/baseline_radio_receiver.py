import numpy as np
import scipy.signal as sig
from commpy.channelcoding import viterbi_decode
from radioml.utils import build_trellis_structure, build_modulator


class Baseline(object):
    """Implementation of traditional radio receiver (baseline) with following modules:
        * MMSE Equalizer
        * Demodulator (QPSK, QAM16, PSK, etc.)
        * Viterbi Decoder

    Arguments:
    ----------

    """
    def __init__(self, modulation_scheme='QPSK'):
        # self.mmse = LeastMeanSquares(
        #     init_params={'equalizer_order':5,
        #                  'random_starts':True,
        #                  'learning_rate':0.01})

        # self.mmse = equalizer
        self.modulator = build_modulator(modulation_scheme)
        self.trellis = build_trellis_structure()

    def __call__(self, convolved_data, convolved_preamble, preamble):

        # Update state
        # @TODO: fix MMSE
        # self.mmse.update(np.squeeze(convolved_preamble),np.squeeze(preamble))
        # data = self.mmse.predict(np.squeeze(convolved_data))
        equalized_data = convolved_preamble
        decoded_bits = self.demodulate(equalized_data)
        estimated_message_bits = self.decode(decoded_bits)

        return estimated_message_bits

    def demodulate(self, inputs):
        return self.modulator.demodulate(inputs, demod_type='hard')

    def decode(self, inputs):
        return viterbi_decode(inputs, self.trellis, 15, 'hard')


# TODO: This is currently not working as expected. Update!!
class MMSEEqualizer(object):
    """Minimum Mean Squared Error Equalizer."""

    def __init__(self, equalizer_order=None, update_rate=0.001):
        if equalizer_order is None:
            raise ValueError("MMSE Equalizer: equalizer_order is missing")
        if equalizer_order % 2 == 0:
            raise ValueError("MMSE Equalizer: equalizer_order must be odd")
        if equalizer_order < 3:
            raise ValueError("MMSE Equalizer: equalizer_order must be at least 3")

        self.order = equalizer_order
        self.L = (self.order - 1) // 2
        self.mu = update_rate
        self.h = np.random.randn(self.order) + 1j * np.random.randn(self.order)

    def update(self, x, y):
        constant = 0j if isinstance(x[0], complex) else 0.0
        x = np.pad(x, self.L, 'constant', constant_values=(constant))
        
        # ###################
        # Close form solution
        # ###################
        A = []
        for i in range(len(y)):
            A += [np.flip(x[i: i + self.order], 0)]
        A = np.array(A)
        self.h, _, _, _ = np.linalg.lstsq(A, y, rcond=-1)
        # ########################
        # Gradient Descent Update
        # ########################
        # b = y
        # for i in range(len(y)):
        #    r = np.flip(x[i: i + self.order],0)
        #    symbol = y[i]
        #    self.h = self.h + self.mu * (symbol - np.dot(r, self.h)) * r.conj()
    def predict(self, x):
        equalized_data = sig.convolve(x, self.h, mode="same")
        return equalized_data