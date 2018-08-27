from radioml.utils import generate_synthetic_packet

def test_packet_generator():
    fake_data = generate_synthetic_packet(5, 10, seed=2018)
    assert fake_data.shape == (15, )