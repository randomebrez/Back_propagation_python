import numpy as np
from Class.Layers import LayerBase


# This Layer transforms 3D images to column
# Input shape : (depth, row, col) => Output shape : (1, depth * row * col)
class FlatLayer(LayerBase.__LayerBase):
    def __init__(self, is_output_layer=False):
        self.parameters = {'input_shape': (0, 0, 0)}
        super().__init__('dense', is_output_layer)

    def initialize(self, input_shape):
        self.parameters['input_shape'] = input_shape
        input_neuron_number = np.product(np.asarray(input_shape))
        self.output_shape = (1, input_neuron_number)
        self.init_cache()

    def init_cache(self):
        return

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        batch_size = inputs.shape[0]
        input_neuron_number = self.output_shape[1]

        activations = inputs.reshape((batch_size, input_neuron_number))
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations
        return activations

    # backward inputs : rows = neuron activation - column = batch index
    def compute_backward(self, inputs):
        input_shape = self.parameters['input_shape']
        batch_size = inputs.shape[0]
        return np.transpose(inputs).reshape((batch_size, input_shape[0], input_shape[1], input_shape[2]))
