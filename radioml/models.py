
"""Contains implementations of Radio Receivers."""
import numpy as np
import scipy.signal as sig
from commpy.modulation import QAMModem
from commpy.channelcoding import viterbi_decode, Trellis


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
    MEMORY = np.array([2])
    G_MATRIX = g_matrix=np.array([[0o7, 0o5]])
    def __init__(self, 
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
        self.mmse = LeastMeanSquares(
            init_params={'equalizer_order':5,
                         'random_starts':True,
                         'learning_rate':0.01})
        self.modulator = self._build_modulator(modulation_scheme)
        self.trellis = Trellis(memory=self.MEMORY, g_matrix=self.G_MATRIX)
        self.tb_depth = tb_depth
        self.decoding_type= decoding_type

    def __call__(self, convolved_data, convolved_preamble, preamble):

        # Update state
        self.mmse.train_closed_form(np.squeeze(convolved_preamble), np.squeeze(preamble))

        equalized_data = self.mmse.predict(np.squeeze(convolved_data))[:-2]

        decoded_bits  = self.demodulate(equalized_data)
        estimated_message_bits = self.decode(decoded_bits)

        return estimated_message_bits

    def equalize(self, inputs):
        raise NotImplementedError
        
    def demodulate(self, inputs):
        return self.modulator.demodulate(inputs, demod_type='hard')

    def decode(self, inputs):
        return viterbi_decode(inputs, self.trellis, self.tb_depth, 
                              self.decoding_type)

    def _build_modulator(self, modulation_scheme):
        """Construct Modulator."""
        if str.lower(modulation_scheme) == 'qpsk':
            return QAMModem(m=4)
        elif str.lower(modulation_scheme) == 'qam16':
            return QAMModem(m=16)
        elif str.lower(modulation_scheme) == 'qam64':
            return QAMModem(m=64)  
        else:
            raise ValueError('Modulation scheme {} is not supported'.format(
                modulation_scheme))


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
        pass


class LeastMeanSquares:
    def __init__(self, init_params={'equalizer_order':None,'random_starts':True, 'learning_rate':0.01}):
        if (not init_params['equalizer_order']):
            raise ValueError("LeastMeanSquares: init_params['equalizer_order'] is missing")
        self.order = init_params['equalizer_order']
        if (self.order % 2 == 0):
            raise ValueError("LeastMeanSquares: init_params['equalizer_order'] must be odd")
        if (self.order < 3):
            raise ValueError("LeastMeanSquares: init_params['equalizer_order'] must be at least 3")
        self.h = None
        self.random_starts = init_params['random_starts']
        self.L = (self.order-1)//2
        self.mu = init_params['learning_rate']
            
    def train_closed_form(self, x, y):
        constant = 0j if isinstance(x[0], complex) else 0.0
        A = []
        x = np.pad(x, self.L, 'constant', constant_values=(constant))
        for i in range(len(y)):
            A += [np.flip(x[i: i+self.order],0)]
        A = np.array(A)
        h,_,_,_ = np.linalg.lstsq(A, y,rcond=-1)
        self.h = h
    
    def predict(self, x):
        if (self.h is None):
            if (self.random_starts):
                self.h = np.random.randn(self.order) + 1j*np.random.randn(self.order) if isinstance(x[0], complex) else np.random.randn(self.order)
            else:
                self.h = np.zeros(self.order, dtype=np.complex_ if isinstance(x[0], complex) else np.float32)
        return sig.convolve(x, self.h , mode="full")[self.L:]