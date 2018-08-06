import random
import math
import multiprocessing as mp
import numpy as np
import commpy as cp
import scipy.signal as sig
from commpy.channelcoding import Trellis
from commpy.modulation import QAMModem


class Radio(object):
    """Simulation of a Radio Transmitter.

    Assumptions:
        * Data is encoded using Convolutional Code.

    Arguments:
    ----------
        data_len:
        preamble_len:
        channels_len:
        modulation_scheme:
        data_rate:
    """
    def __init__(self, data_len, preamble_len, channels_len, 
                 modulation_scheme='qpsk', data_rate=1/2):
        self.modulator = self._build_modulator(modulation_scheme)
        self.data_len = data_len
        self.preamble_len = preamble_len
        self.channels_len = channels_len
        self.data_rate = data_rate
        self.trellis = Trellis(memory=np.array([2]), 
                              g_matrix=np.array([[0o7, 0o5]]))

    def emit_signal(self, seed=None):
        """Simulate data from a transmitter"""
        
        # Generate preamble and message bits
        np.random.seed(seed)
        preamble       = np.random.randint(0, 2, self.preamble_len)
        message_bits   = np.random.randint(0, 2, self.data_len)
        packet         = np.concatenate([preamble, message_bits])

        # Simulate TX
        encoded_packet   = cp.channelcoding.conv_encode(packet, self.trellis)
        encoded_packet   = encoded_packet[:-2*int(self.trellis.total_memory)]
        modulated_packet = self.modulator.modulate(encoded_packet)

        return (packet, modulated_packet)

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


class RadioDataGenerator(Radio):
    def __init__(self, data_len, preamble_len, channels_len, modulation_scheme='qpsk', data_rate=1/2):
        super(RadioDataGenerator, self).__init__(data_len, preamble_len, 
                                                 channels_len, 
                                                 modulation_scheme, 
                                                 data_rate)
                            
    def _channel_interefence(self, inputs):
        """Simulate multi-tap channel interference.
        
        Arguments:
        ----------
            inputs: complex ndarray: [batch, data_length]

        Returns:
        --------
            convolved_inputs: complex ndarray
        """
        # @TODO : assert input shape validataion
        x = np.random.uniform(-1, 1, (len(inputs), self.channels_len))  
        channels = x / np.linalg.norm(x, axis=-1)[:, np.newaxis]
        
        # @TODO : vectorize this op
        a = [sig.convolve(x, y, mode='same') for x, y in zip(inputs, channels)]
        return np.array(a), channels

    def _carrier_frequency_offset(self, inputs, omegas):
        """Simulate Carrier frequency offset (CFO) @ some omega.

        Argument:
        ---------
            inputs: complex ndarray: [batch, data_length]
            omegas: float array [1, batch]
        """
        # @TODO : assert input shape validataion

        # Example: 
        #   If `batch_size` = 2, `data_len`` = 3 --> time_steps_matrix = 
        # [[0, 1, 2],
        #  [0, 1, 2]]
        batch_size, data_len = np.shape(inputs)
        time_steps_matrix = np.tile(np.arange(data_len),(batch_size, 1))
        rotated_packets = inputs * np.exp(1j * omegas  * time_steps_matrix)
        return rotated_packets

       
    def _data_genenerator(self, transform_func, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """A generic generator returns an `Iterator` object that 
        generates (inputs, labels) until it raises a `StopIteration` exception, 
        
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
                signals = pool.map(self.emit_signal,[(seed + i if seed else None) 
                                      for i in range(batch_size)])
                inputs, labels = transform_func(signals, omega, snr_dB, seed)
                yield inputs, labels
        except Exception as e:
            print(e)
            raise e
        finally:
            pool.close()

    def cfo_data_generator(self, omega, snr_dB, batch_size, 
                          seed=None, num_cpus=4):
        """
        Returns: `Iterator` object that generates (inputs, outputs) as 
            Inputs: [preamble, cfo_preamble]
            Outputs: cfo_corrected_preamble
        """
        def _cfo_data_func(dataset, omega, snr_dB, seed=None):
            np.random.seed(seed)
            # Unpack  radio data
            _, modulated_packets = zip(*dataset)
            batch_size = len(modulated_packets)

            modulated_packets = np.array(modulated_packets)
            # Add AWGN noise
            noisy_packets = cp.channels.awgn(modulated_packets.flatten(), snr_dB) 
            noisy_packets = noisy_packets.reshape((batch_size, -1))

            # Simulate CFO
            w_batch = np.random.uniform(-omega, omega, size=(batch_size, 1))
            rotated = self._carrier_frequency_offset(noisy_packets, w_batch)
            
            # Process Inputs
            preambles = modulated_packets[:, :self.preamble_len]
            preambles = self._encode_complex_to_real(preambles)
            preambles_conv = self._encode_complex_to_real(rotated[:, :self.preamble_len])

            # Process labels
            cfo_corrected_preamble = noisy_packets[:, :self.preamble_len]
            cfo_corrected_preamble = self._encode_complex_to_real(cfo_corrected_preamble)
        
            return [preambles, preambles_conv], cfo_corrected_preamble

        return self._data_genenerator(_cfo_data_func,
                                     omega, snr_dB, batch_size, seed, num_cpus)


    def equalization_data_generator(self,omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """ 
        Returns: `Iterator` object that generates (inputs, outputs) as 
            Inputs: [preamble, cfo_corected_preamble, cfo_corrected_data], 
            Outputs: equalized_packet
        """
        def _process_data_for_equalization_net(dataset, omega, snr_dB, seed=None):
            np.random.seed(seed)

            # Unpack  radio data
            _, modulated_packets = zip(*dataset)
    
            modulated_packets = np.array(modulated_packets)
            batch_size = len(modulated_packets)

            preambles = modulated_packets[:, : self.preamble_len]
            preambles = self._encode_complex_to_real(preambles)

            # Add AWGN noise
            noisy_packets = cp.channels.awgn(modulated_packets.flatten(), snr_dB) 
            noisy_packets = noisy_packets.reshape((batch_size, -1))

            # Simulate multi-tap channel interference
            convolved_packets, _ = self._channel_interefence(noisy_packets)

            ################
            # Process Inputs
            ################
            cfo_corrected_preamble = convolved_packets[:, :self.preamble_len]
            cfo_corrected_data     = convolved_packets[:, self.preamble_len:]

            cfo_corrected_preamble = self._encode_complex_to_real(cfo_corrected_preamble)
            cfo_corrected_data = self._encode_complex_to_real(cfo_corrected_data)
            
            # Process Label
            x = noisy_packets[:, self.preamble_len:]
            equalized_packet = self._encode_complex_to_real(x)
    
            return [preambles, cfo_corrected_preamble, cfo_corrected_data],\
                    [equalized_packet, modulated_packets[:, self.preamble_len:]]

        return self._data_genenerator(_process_data_for_equalization_net,
                                      omega, snr_dB, batch_size, seed, num_cpus)


    def ecc_data_generator(self, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """
        Returns: `Iterator` object that generates (inputs, outputs) as 
            Inputs: equalized_packet
            Outputs: message_bits
        """

        def _process_data_for_demod_n_ecc_net(dataset, omega, snr_dB, seed=None):
            np.random.seed(seed)

            # Unpack  radio data
            original_packet, modulated_packets = zip(*dataset)
            
            # Process inputs
            x = np.array(modulated_packets)[:,self.preamble_len:]
            noisy = cp.channels.awgn(x.flatten(), snr_dB).reshape((len(x), -1))
            equalized_packet = self._encode_complex_to_real(noisy)

            # Process labels:
            message_bits = np.array(original_packet)[:, self.preamble_len:] 
            message_bits = np.expand_dims(message_bits, -1)

            vis = self._encode_complex_to_real(x)
            return [equalized_packet, vis], message_bits

        return self._data_genenerator(_process_data_for_demod_n_ecc_net,
                                      omega, snr_dB, batch_size, seed, num_cpus)

    def end2end_data_generator(self,omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """
        Returns: `Iterator` object that generates (inputs, outputs) as 

            Inputs: [preamble, corrupted_packet], 
            Outputs: original_message_bits
        """
        def _process_data_end2end_net(dataset, omega, snr_dB, seed=None):
            np.random.seed(seed)
            
            # Unpack  radio data
            original_packets, modulated_packets = zip(*dataset)
            batch_size = len(modulated_packets)

            # Simulate multi-tap channel interference
            convolved_packets, channels = self._channel_interefence(modulated_packets)

            # Add AWGN noise
            noisy_packets = cp.channels.awgn(convolved_packets.flatten(), snr_dB) 
            noisy_packets = noisy_packets.reshape((batch_size, -1))

            # Simulate CFO
            w_batch = np.random.uniform(-omega, omega, size=(batch_size, 1))
            rotated = self._carrier_frequency_offset(noisy_packets, w_batch)

            # Process inputs
            preambles = np.array(modulated_packets)[:, :self.preamble_len]        
            preambles = self._encode_complex_to_real(preambles)
            corrupted_packets = self._encode_complex_to_real(rotated)

            # Process labels
            orignal_message_bits = np.expand_dims(
                np.array(original_packets)[:, self.preamble_len:], -1)
                
            return [preambles, corrupted_packets], \
                    [orignal_message_bits, w_batch, channels]

        return self._data_genenerator(_process_data_end2end_net,
                                      omega, snr_dB, batch_size, seed, num_cpus)


    def _encode_complex_to_real(self, complex_inputs):
        """TF does not support complex numbers for training. 
        We encode complex inputs into 2D array."""
        return np.stack([np.real(complex_inputs), np.imag(complex_inputs)], -1)