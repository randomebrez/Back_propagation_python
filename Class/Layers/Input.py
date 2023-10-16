from Class.Layers import LayerBase


class InputLayer(LayerBase.__LayerBase):
    def __init__(self, input_shape):
        super().__init__('input', False)
        self.output_shape = input_shape
        self.output_dimension = len(input_shape)

    def compute(self, inputs, store):
        if store:
            self.cache['activation_values'] = inputs
        return inputs
