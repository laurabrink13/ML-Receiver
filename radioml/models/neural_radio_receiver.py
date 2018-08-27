import numpy as np


class NeuralRadioReceiver(object):
    """Neural Defined Radio Receiver.

    Arguments:
    ----------

    """
    def __init__(self, data_length, pre_trained_model_path=None):
        super(NeuralRadioReceiver, self).__init__()

        self.data_length = data_length
        self.model = self._build_model(pre_trained_model_path)

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

    def _build_model(self, pre_trained_model_path):
        pass
