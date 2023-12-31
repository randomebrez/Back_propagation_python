import numpy as np
from Class.Layers import LayerBase
import Tools.ConvolutionHelper as helper
import time

# Input & Output shape : (depth, row, col)
class TransposedConvolutionLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, zero_padding, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True):
        self.input_reshaped_shape = (0, 0, 0)
        self.filters = np.array([])
        self.mirror_filters = np.array([])
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': [kernel_size, kernel_size], 'stride': [stride, stride], 'zero_padding': [zero_padding, zero_padding], 'use_bias': use_bias}
        self.use_bias = use_bias
        self.biases = np.array([])
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        super().__init__('transposed_convolution', is_output_layer)

    def initialize(self, input_shape):
        input_depth = input_shape[0]
        self.parameters['input_shape'] = input_shape

        kernel_size = self.parameters['kernel_size']
        filter_number = self.parameters['filter_number']

        self.output_shape, self.input_reshaped_shape = self.shapes_init(input_shape)

        self.filters = np.random.rand(input_depth * filter_number, kernel_size[0], kernel_size[1])
        if self.parameters['use_bias']:
            self.biases = np.zeros((self.parameters['filter_number'], 1, 1))

        self.init_cache()

    def shapes_init(self, input_shape):
        input_depth = input_shape[0]
        (kernel_size_x, kernel_size_y) = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']

        reshaped_input_x = 2 * (kernel_size_x - zero_padding[0] - 1) + input_shape[1] + (input_shape[1] - 1) * (stride[0] - 1)
        reshaped_input_y = 2 * (kernel_size_y - zero_padding[1] - 1) + input_shape[2] + (input_shape[2] - 1) * (stride[1] - 1)
        shaped_input_shape = (input_depth, reshaped_input_x, reshaped_input_y)

        x_jump = (reshaped_input_x - kernel_size_x + 1)
        y_jump = (reshaped_input_y - kernel_size_y + 1)
        output_shape = (self.parameters['filter_number'], x_jump, y_jump)

        return output_shape, shaped_input_shape

    def init_cache(self):
        self.cache['sigma_primes'] = []
        self.cache['reshaped_inputs'] = np.zeros((self.input_reshaped_shape))
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        return self.patch_compute(inputs, store)

    def compute_backward_and_update_weights(self, bp_inputs, learning_rate):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']

        sigma_primes = self.cache['sigma_primes']
        reshaped_inputs = self.cache['reshaped_inputs']

        back_activation_values = sigma_primes * bp_inputs

        batch_size = back_activation_values.shape[0]
        bp_outputs = np.zeros((batch_size,) + input_shape)
        dE_dw = np.zeros(self.filters.shape)

        for filter_index in range(filter_number):
            for patch, x, y in self.patches_generator(bp_outputs, kernel_size):
                f_min = filter_index * input_shape[0]
                temp = back_activation_values[:, filter_index, x, y]
                patch += self.filters[f_min:f_min + input_shape[0]] * temp[:, np.newaxis, np.newaxis, np.newaxis]

            for patch, x, y in self.patches_generator(reshaped_inputs, kernel_size):
                f_min = filter_index * input_shape[0]
                dE_dw[f_min:f_min + input_shape[0]] += np.sum(patch * back_activation_values[:, filter_index, x, y].reshape((batch_size, 1, 1, 1)), axis=0)

        self.filters -= learning_rate * dE_dw / batch_size

        if self.use_bias:
            bias_variation = learning_rate * np.mean(back_activation_values, axis=(0, 2, 3), keepdims=True).reshape((filter_number, 1, 1))
            self.biases -= bias_variation

        return bp_outputs

    def patch_compute(self, inputs, store):
        kernel_size = self.parameters['kernel_size']
        stride = self.parameters['stride']
        zero_padding = self.parameters['zero_padding']
        input_shape = self.parameters['input_shape']
        reshape_inputs_shape = self.input_reshaped_shape
        filter_number = self.parameters['filter_number']
        batch_size = inputs.shape[0]

        reshape_inputs = helper.input_transposed_convolution_shape(batch_size, inputs, kernel_size, stride, zero_padding, input_shape)

        tick = time.time()
        feature_maps = np.zeros((batch_size, filter_number)  + self.output_shape[1:])
        for patch, x, y in self.patches_generator(reshape_inputs, kernel_size):
            for filter_index in range(filter_number):
                f_min = filter_index * reshape_inputs_shape[0]
                product = patch * self.filters[f_min:f_min+reshape_inputs_shape[0]]
                conv_result = np.sum(product, axis=(1, 2, 3)).reshape((batch_size, 1))
                feature_maps[:, f_min:f_min+reshape_inputs_shape[0], x, y] += conv_result

        #print("patch computation : {0}".format(round(time.time() - tick, 2)))

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(feature_maps)
            self.cache['sigma_primes'] = sigma_primes
            self.cache['reshaped_inputs'] = reshape_inputs
        else:
            activations = self.activation_function(feature_maps)

        # Apply bias
        if self.parameters['use_bias']:
            activations += self.biases

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        # print("Conv FF exec time : {0}".format(round(time.time() - tick, 2)))
        return activations

    # Inputs are shaped entering the layer.
    # Patches generation is then with no padding and a stride of 1
    def patches_generator(self, inputs, kernel_size):
        input_x, input_y = inputs.shape[2], inputs.shape[3]
        kernel_size_x, kernel_size_y = kernel_size[0], kernel_size[1]

        for x in range(input_x - kernel_size_x + 1):
            for y in range(input_y - kernel_size_y + 1):
                patch = inputs[:, :, x:x+kernel_size_x, y:y+kernel_size_y]
                yield patch, x, y
