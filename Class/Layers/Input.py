import pandas as pd
from Class.Layers import LayerBase


class InputLayer(LayerBase.__LayerBase):
    def __init__(self, layer_size):
        super().__init__('input', layer_size, False)

    def compute(self, inputs, store):
        if store:
            self.cache['activation_values'] = inputs
        return inputs
