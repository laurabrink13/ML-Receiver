import math
import multiprocessing as mp
import numpy as np
import commpy as cp
import scipy.signal as sig
from commpy.channelcoding import Trellis
from commpy.modulation import QAMModem

class RadioData(object):
    """Simulation of a Radio Transmitter that sends data over AWGN Channels.
    
        Assumptions:
            * Data is encoded using Convolutional Code.

    Arguments:
    ----------
        data_len:
        preamble_len:
        channels_len:
        modulation_scheme:
        data_rate
        

    """
    MEMORY = np.array([2])
    G_MATRIX = g_matrix=np.array([[0o7, 0o5]])
    def __init__(self, 
                 data_len, 
                 preamble_len, 
                 channels_len, 
                 modulation_scheme='qpsk', 
                 data_rate=1/2):
        self.modulator = self._build_modulator(modulation_scheme)
        self.trellis = Trellis(memory=self.MEMORY, g_matrix=self.G_MATRIX)
        self.data_len = data_len
        self.preamble_len = preamble_len
        self.channels_len = channels_len
        self.data_rate = data_rate

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

    def generate_packet(self, 
                        omega = 1/100,
                        snr_dB=15.0):
        """Simulate data over AWGN Channel."""
        
        # Generate preamble and message bits
        preamble       = np.random.randint(0, 2, self.preamble_len)
        message_bits   = np.random.randint(0, 2, self.data_len)

        # shape: [preamble_len + data_len, 1]
        packet         = np.concatenate([preamble, message_bits])

        # Simulate TX

        # shape: [2*(preamble_len + datal_len) + 4, 1]
        encoded_packet   = cp.channelcoding.conv_encode(packet, self.trellis)[:-4]

        # shape: [preamble_len + data_len + 2, 1]
        modulated_packet = self.modulator.modulate(encoded_packet)

        # Simulate multi-tap channel interference
        channels = np.random.uniform(0, 1, self.channels_len)  # -1 to 1
        channels = channels / channels.sum()  # normalize to sum of one
        convolved_packet = sig.convolve(modulated_packet, channels, mode='same')

        # Simulate Carrier frequency offset (CFO) @ some omega
        w = np.random.uniform(low=-omega, high=omega)

        cfo = np.exp(1j *  w * np.arange(len(convolved_packet)))
        rotated_packet = convolved_packet * cfo

        # Simulate packet sending over AWGN channel @ some signal-to-noise ratio
        corrupted_packet = cp.channels.awgn(rotated_packet, snr_dB)

        return (packet, w,
               modulated_packet,
               convolved_packet, 
               rotated_packet, 
               corrupted_packet)


    def cfo_correction_data_gen(self, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """Generate cfo data correction."""
        pool = mp.Pool(num_cpus)
        try:
            np.random.seed(seed)
            while True:
                batch = pool.starmap(self.generate_packet, 
                                    [(omega, snr_dB) for i in range(batch_size)])
                np.random.seed()

                # In order to train CFO Correction Network, we only need access to 
                # omegas, preambles and preambles convolved data.
                _, omegas, modulated_packets, _, _, corrupted_packets = zip(*batch)

                # Obtain the preamble and preamble conv
                preambles     = self._encode_complex(np.array(modulated_packets)[:,:self.preamble_len])
                preamble_conv = self._encode_complex(np.array(corrupted_packets)[:, :self.preamble_len])

                yield [preambles, preamble_conv], np.array(omegas)
        except Exception as e:
            print(e)
        finally:
            pool.close()

    def equalize_data_gen(self, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """Generate data for Equalization Net."""
        pool = mp.Pool(num_cpus)
        try:
            np.random.seed(seed)
            while True:
                batch = pool.starmap(self.generate_packet, 
                                    [(omega, snr_dB) for i in range(batch_size)])
                np.random.seed()

                # In order to train CFO Correction Network, we only need access to 
                # omegas, preambles and preambles convolved data.
                _, omegas, modulated_packets, _, _, corrupted_packets = zip(*batch)

                # Obtain the preamble and preamble conv
                preambles     = self._encode_complex(np.array(modulated_packets)[:,:self.preamble_len])
                preamble_conv = self._encode_complex(np.array(corrupted_packets)[:, :self.preamble_len])

                yield [preambles, preamble_conv], np.array(omegas)
        except Exception as e:
            print(e)
        finally:
            pool.close()


    def end2end_data_generator(self, omega, snr_dB, batch_size, seed=None, num_cpus=4):
        """Generate  data for end 2 end training."""
        pool = mp.Pool(num_cpus)
        try:
            np.random.seed(seed)
            while True:
                batch = pool.starmap(self.generate_packet, 
                                    [(omega, snr_dB) for i in range(batch_size)])
                np.random.seed()        

                packets, _, modulated_packets, _, _, corrupted_packets = zip(*batch)
                
                packets = np.expand_dims(np.array(packets), -1)

                # Obtain the preamble and preamble conv
                preamble = self._encode_complex(np.array(modulated_packets)[:, :self.preamble_len])
                preamble_conv = self._encode_complex(np.array(corrupted_packets)[:, :self.preamble_len])
                corrupted_packets = self._encode_complex(np.array(corrupted_packets))

                yield [corrupted_packets, preamble, preamble_conv], packets
        except Exception as e:
            print(e)
        finally:
            pool.close()  

    def _encode_complex(self, complex_inputs):
        """TF does not support complex numbers for training. 
        We encode complex inputs into 2D array."""
        return np.stack([np.real(complex_inputs), np.imag(complex_inputs)], -1)