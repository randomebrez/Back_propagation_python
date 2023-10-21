import numpy as np
from Class.Layers import LayerBase


# weight matrix of a one to one layer is Id.
# FF computation : Act_fct(inputs)
# BP computation : Dot product : Grad(Act_fct) . BP_inputs
class OneToOneLayer(LayerBase.__LayerBase):
    def __init__(self, activation_function, activation_function_with_derivative, is_output_layer=False):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_with_derivative
        super().__init__('one_to_one', is_output_layer)

    def initialize(self, input_shape):
        self.output_shape = input_shape

    def compute(self, inputs, store):
        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_derivative(inputs)
            self.cache['sigma_primes'] = sigma_primes
        else:
            activations = self.activation_function(inputs)

        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def compute_backward_and_update_weights(self, bp_inputs, learning_rate):
        # Compute previous layer's inputs
        sigma_primes = self.cache['sigma_primes']
        if len(sigma_primes.shape) == 2:
            return sigma_primes * bp_inputs
        else:
            result = np.zeros(bp_inputs.shape)
            for i in range(bp_inputs.shape[0]):
                result[i] = np.dot(sigma_primes[i], bp_inputs[i])
        return result
