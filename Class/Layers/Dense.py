import numpy as np
from Class.Layers import LayerBase


class DenseLayer(LayerBase.__LayerBase):
    def __init__(self, layer_size, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True, normalization_function=None):
        self.weight_matrix = []
        self.biases = np.zeros((1, layer_size))
        self.use_bias = use_bias
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        self.normalization = normalization_function
        super().__init__('dense', is_output_layer)
        self.output_shape = (1, layer_size, 1)

    def initialize(self, input_shape):
        input_number = np.product(np.asarray(input_shape))
        self.weight_matrix = 0.01 * np.random.rand(input_number, self.output_shape[1])
        self.init_cache()

    def init_cache(self):
        self.cache['sigma_primes'] = []
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        # if inputs are not organized as (batch_index, input_values) reshape them to get row wise inputs
        shaped_inputs = self.shape_input(inputs)

        # Aggregate inputs, weights, and biases
        aggregation_result = np.dot(shaped_inputs, self.weight_matrix) + self.biases

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(aggregation_result)
            self.cache['sigma_primes'] = sigma_primes
        else:
            activations = self.activation_function(aggregation_result)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def compute_backward(self, inputs):
        back_activation_values = self.cache['sigma_primes'] * inputs

        if self.normalization is not None:
            back_activation_values = self.normalization(back_activation_values)

        self.cache['back_activation_values'] = back_activation_values

        # Compute previous layer's inputs
        return np.dot(back_activation_values, np.transpose(self.weight_matrix))

    def update_weights(self, previous_layer_activation, learning_rate):
        current_layer_activations = self.cache['back_activation_values']
        sample_number = np.shape(current_layer_activations)[1]

        # Update Bias
        if self.use_bias:
            bias_variation = (learning_rate / sample_number) * np.sum(current_layer_activations, axis=0, keepdims=True)
            self.biases -= bias_variation

        # Update weight
        weight_gradient = np.dot(np.transpose(previous_layer_activation), current_layer_activations)
        self.weight_matrix -= (learning_rate / sample_number) * weight_gradient

    def shape_input(self, inputs):
        if len(inputs.shape) < 2:
            return inputs

        batch_number = inputs.shape[0]
        item_input_number = np.product(inputs.shape[1:])
        return inputs.reshape((batch_number, item_input_number))

    def get_back_activation_values(self):
        return self.cache['back_activation_values']

    def get_sigma_prime_values(self):
        return self.cache['sigma_primes']
