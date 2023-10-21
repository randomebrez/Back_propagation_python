class __LayerBase:
    def __init__(self, layer_type, is_output_layer=False):
        self.layer_type = layer_type
        self.output_shape = (1, 1, 1)
        self.is_output_layer = is_output_layer
        self.cache = {'activation_values': []}

    def initialize(self, input_shape):
        return

    def clean_cache(self):
        self.cache = {'activation_values': [], 'inputs': []}

    def update_weights(self, previous_layer_activation, learning_rate):
        return

    def get_activation_values(self):
        return self.cache['activation_values']
