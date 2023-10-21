import numpy as np
from Class.Layers import LayerBase


class DenseLayer(LayerBase.__LayerBase):
    def __init__(self, layer_size, is_output_layer=False, use_bias=True):
        self.weight_matrix = []
        self.biases = np.zeros((1, layer_size))
        self.use_bias = use_bias
        super().__init__('dense', is_output_layer)
        self.output_shape = (layer_size,)

    def initialize(self, input_shape):
        self.weight_matrix = 0.001 * np.random.rand(input_shape[0], self.output_shape[0])
        self.init_cache()

    def init_cache(self):
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        # Aggregate inputs, weights, and biases
        aggregation_result = np.dot(inputs, self.weight_matrix) + self.biases

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['inputs'] = inputs
            self.cache['activation_values'] = aggregation_result

        return aggregation_result

    def compute_backward_and_update_weights(self, bp_inputs, learning_rate):
        inputs = self.cache['inputs']
        sample_number = np.shape(bp_inputs)[0]

        # Compute next_layer_bp_inputs
        next_layer_bp_inputs = np.dot(bp_inputs, np.transpose(self.weight_matrix))

        # Update Bias
        if self.use_bias:
            bias_variation = (learning_rate / sample_number) * np.sum(bp_inputs, axis=0, keepdims=True)
            self.biases -= bias_variation

        # Update weight
        weight_gradient = np.dot(np.transpose(inputs), bp_inputs)
        self.weight_matrix -= (learning_rate / sample_number) * weight_gradient

        # Compute previous layer's inputs
        return next_layer_bp_inputs
