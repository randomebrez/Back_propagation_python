from Class.Layers import LayerBase


class InputLayer(LayerBase.__LayerBase):
    def __init__(self, layer_size):
        self.output_shape = (layer_size, 0, 0)
        super().__init__('input', False)

    def compute(self, inputs, store):
        if store:
            self.cache['activation_values'] = inputs
        return inputs
