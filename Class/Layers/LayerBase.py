class __LayerBase:
    def __init__(self, layer_type, layer_size, is_output_layer=False):
        self.layer_size = layer_size
        self.layer_type = layer_type
        self.is_output_layer = is_output_layer
        self.cache = {'activation_values': []}

    def initialize(self, previous_layer_size=0):
        return

    def clean_cache(self):
        self.cache = {'activation_values': []}

    def update_weights(self, previous_layer_activation, learning_rate):
        return

    def get_activation_values(self):
        return self.cache['activation_values']
