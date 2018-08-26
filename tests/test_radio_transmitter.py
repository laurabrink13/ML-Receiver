import numpy as np
from collections import namedtuple
from radioml.core import RadioTransmitter
from radioml.utils import build_trellis_structure
from commpy.channelcoding.convcode import viterbi_decode


def test_preamble_length_after_modulation():
    TestModulator = namedtuple('TestModulator', ['mod_scheme', 'len', 'expected_length'])
    tests = [
        TestModulator('QPSK', 90, 90),
        TestModulator('QAM16',90, 45),
        TestModulator('QAM64',90, 30),
    ]
    for test in tests:
        # Create a fake packet
        packet = np.zeros(shape=(test.len, ))
        tx = RadioTransmitter(0, len(packet), modulation_scheme=test.mod_scheme)
        _, got = tx._simulate_radio_transmitter(packet)
        assert len(got[:tx.modulated_preamble_len]) == test.expected_length


def test_radio_encoder():
    """Online tool to visualize Encoder:
    http://www.ee.unb.ca/cgi-bin/tervo/viterbi.pl
    """
    # Create a fake packet
    packet = [0, 0, 1, 1]
    # Test convolution code encoder at different memory size 
    EncoderTest = namedtuple('EncoderTest', ['trellis', 'wanted'])
    tests = [
        EncoderTest(None,  # default option (r= 1/2, k=3)  
                    np.array([0, 0, 0, 0, 1, 1, 0, 1])),
        EncoderTest(build_trellis_structure(1, 2, constraint_len=4),
                    np.array([0, 0, 0, 0, 1, 1, 0, 1])),
        EncoderTest(build_trellis_structure(1, 3, constraint_len=4), 
                    np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0 ,0])),
    ]
    for test in tests:
        tx = RadioTransmitter(0, 0, 'qpsk', test.trellis)
        # Test correctness of encoder
        got, _ = tx._simulate_radio_transmitter(packet)
        np.testing.assert_array_equal(got, test.wanted)

        # Test correctness of output message by decoding with Viterbi
        decoded = viterbi_decode(got, tx.trellis, 4)
        np.testing.assert_array_equal(packet, decoded)

