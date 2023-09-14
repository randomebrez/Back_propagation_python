from Class.Layers import LayerBase


# weight matrix of a one to one layer is Id.
# FF computation : activate inputs
# BP computation : multiply each input by corresponding sigma_prime and return
# May consider to update weights. For now all are one and remain unchanged
class OneToOneLayer(LayerBase.__LayerBase):
    def __init__(self, activation_function, is_output_layer=False):
        self.activation_function = activation_function
        super().__init__('one_to_one', is_output_layer)

    def initialize(self, input_shape):
        self.output_shape = input_shape

    def compute(self, inputs, store):
        return self.activation_function(inputs)

    def compute_backward(self, inputs):
        return inputs
