from Class.Layers import LayerBase

class NormalizationLayer(LayerBase.__LayerBase):
    def __init__(self, activation_function, is_output_layer=False):
        self.activation_function = activation_function
        super().__init__('normalization', is_output_layer)

    def initialize(self, input_shape):
        self.output_shape = input_shape

    def compute(self, inputs, store):
        activations = inputs # self.activation_function(inputs)

        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def compute_backward(self, inputs):
        return self.activation_function(inputs)