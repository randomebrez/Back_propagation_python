from Class.Layers import LayerBase


# This Layer transforms 3D images to column
# Input shape : (depth, row, col) => Output shape : (1, depth * row * col)
class ReshapeLayer(LayerBase.__LayerBase):
    def __init__(self, output_shape, is_output_layer=False):
        self.parameters = {'input_shape': (0, 0, 0)}
        super().__init__('reshape', is_output_layer)
        self.output_shape = output_shape

    def initialize(self, input_shape):
        self.parameters['input_shape'] = input_shape
        self.init_cache()

    def init_cache(self):
        return

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        batch_size = inputs.shape[0]
        output_shape = (batch_size,) + self.output_shape

        activations = inputs.reshape(output_shape)
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations
        return activations

    # backward inputs : rows = neuron activation - column = batch index
    def compute_backward(self, inputs):
        input_shape = self.parameters['input_shape']
        batch_size = inputs.shape[0]
        reshaped_inputs = inputs.reshape((batch_size,) + input_shape)
        return reshaped_inputs
