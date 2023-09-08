import numpy as np
from Class.Layers import LayerBase


class DenseLayer(LayerBase.__LayerBase):
    def __init__(self, layer_size, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True):
        self.weight_matrix = []
        self.biases = np.zeros((layer_size, 1))
        self.use_bias = use_bias
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        super().__init__('dense', layer_size, is_output_layer)

    def initialize(self, previous_layer_size=0):
        self.weight_matrix = 0.01 * np.random.rand(self.layer_size, previous_layer_size)
        self.cache['sigma_primes'] = []
        self.cache['back_activation_values'] = []

    def compute(self, inputs, store):
        aggregation_result = np.dot(self.weight_matrix, inputs) + self.biases
        if store or self.is_output_layer:
            activations, sigma_primes = self.activation_function_with_derivative(aggregation_result)
            self.cache['activation_values'], self.cache['sigma_primes'] = activations, sigma_primes
        else:
            activations = self.activation_function(aggregation_result)
        return activations

    def compute_backward(self, inputs):
        back_activation_values = self.cache['sigma_primes'] * inputs
        self.cache['back_activation_values'] = back_activation_values

        # Compute previous layer's inputs
        return np.dot(np.transpose(self.weight_matrix), back_activation_values)

    def update_weights(self, previous_layer_activation, learning_rate):
        current_layer_activations = self.cache['back_activation_values']
        sample_number = np.shape(current_layer_activations)[1]

        # Update Bias
        if self.use_bias:
            bias_variation = (learning_rate / sample_number) * np.sum(current_layer_activations, axis=1, keepdims=True)
            self.biases -= bias_variation

        # Update weight
        weight_gradient = np.dot(current_layer_activations, np.transpose(previous_layer_activation))
        self.weight_matrix -= (learning_rate / sample_number) * weight_gradient

    def get_back_activation_values(self):
        return self.cache['back_activation_values']

    def get_sigma_prime_values(self):
        return self.cache['sigma_primes']
