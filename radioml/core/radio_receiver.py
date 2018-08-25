
class RadioReceiver(object):
    """Abstract Radio Receiver."""
    def __init__(self):
        pass

    def __call__(self, noisy_signals, **kwargs):
        """Estimate noisy signals

        Arguments:
            noisy_signals (complex/float ndarray):

        Return:
            data_estimate (int ndarray): estimate of original message bits
        """
        raise NotImplementedError
