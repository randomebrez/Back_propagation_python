from Class.Layers import LayerBase


class InputLayer(LayerBase.__LayerBase):
    def __init__(self, layer_size):
        super().__init__('input', False)
        self.output_shape = (1, layer_size, 1)

    def compute(self, inputs, store):
        if store:
            self.cache['activation_values'] = inputs
        return inputs
