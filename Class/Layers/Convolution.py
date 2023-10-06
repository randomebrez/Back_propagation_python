import numpy as np
from Class.Layers import LayerBase
import Tools.ConvolutionHelper as helper


# Input & Output shape : (depth, row, col)
class ConvolutionFFTLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True, normalization_function=None):
        self.filters = []
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': [0, kernel_size, kernel_size], 'stride': [0, stride, stride], 'zero_padding': [0, 0, 0], 'use_bias': use_bias}
        self.biases = []
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        self.normalization = normalization_function
        super().__init__('convolution', is_output_layer)

    def initialize(self, input_shape):
        input_depth = input_shape[0]
        self.parameters['input_shape'] = input_shape
        self.parameters['zero_padding'][1:], self.output_shape = self.padding_and_output_shape_compute()

        # Set depth wise parameters already known by input shape
        self.parameters['kernel_size'][0] = input_depth
        self.parameters['zero_padding'][0] = input_depth

        kernel_size = self.parameters['kernel_size']
        filter_number = self.parameters['filter_number']

        self.filters = 0.01 * np.random.rand(input_depth * filter_number, kernel_size[1], kernel_size[2])

        if self.parameters['use_bias']:
            self.biases = np.random.rand(self.parameters['filter_number'], 1, 1)

        self.init_cache()

    def padding_and_output_shape_compute(self):
        input_shape = self.parameters['input_shape']
        (_, kernel_size_x, kernel_size_y) = self.parameters['kernel_size']
        (_, stride_x, stride_y) = self.parameters['stride']

        input_shape_x = input_shape[1]
        input_shape_y = input_shape[2]

        x_jump = (input_shape_x - kernel_size_x + 1) // stride_x
        x_padding_needed = kernel_size_x + stride_x * x_jump - input_shape_x
        x_padding_each_side = (x_padding_needed + 1) // 2

        if input_shape_x != input_shape_y or kernel_size_x != kernel_size_y or stride_y != stride_x:
            y_jump = (input_shape_y - kernel_size_y + 1) // stride_y
            y_padding_needed = kernel_size_y + stride_y * y_jump - input_shape_y
            y_padding_each_side = (y_padding_needed + 1) // 2

            output_shape = (self.parameters['filter_number'], x_jump + 1, y_jump + 1)

            return (x_padding_each_side, y_padding_each_side), output_shape
        else:
            output_shape = (self.parameters['filter_number'], x_jump + 1, x_jump + 1)

            return (x_padding_each_side, x_padding_each_side), output_shape

    def init_cache(self):
        self.cache['sigma_primes'] = []
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        kernel_size = self.parameters['kernel_size']
        stride = self.parameters['stride']
        zero_padding = self.parameters['zero_padding']
        input_shape = self.parameters['input_shape']

        batch_size = inputs.shape[0]

        # Shape inputs to effective convolution
        shaped_inputs = helper.input_compute_shape(inputs, kernel_size[1])

        # Convolve
        batch_feature_maps = helper.convolve_filters(shaped_inputs, self.filters)

        # Shape convolution result according to stride & padding
        shaped_feature_maps = helper.input_conv_res_shape(batch_feature_maps, batch_size, kernel_size, stride, zero_padding, input_shape, self.output_shape)

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(shaped_feature_maps)
            self.cache['sigma_primes'] = sigma_primes
        else:
            activations = self.activation_function(shaped_feature_maps)

        # Apply bias
        if self.parameters['use_bias']:
            for i in range(batch_size):
                shaped_feature_maps[i] += self.biases

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def compute_backward(self, inputs):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']

        # compute back_activation values
        back_activation_values = self.cache['sigma_primes'] * inputs

        if self.normalization is not None:
            back_activation_values = self.normalization(back_activation_values)

        # Save to update weights
        self.cache['back_activation_values'] = back_activation_values

        # Shape filters and BP activated values for FFT convolution
        filter_mirror, shaped_bp_values = helper.de_compute_shape(self.filters, back_activation_values, filter_number, kernel_size, stride, input_shape)

        # FFT conv
        conv_res = helper.convolve_filters(filter_mirror, shaped_bp_values)

        # Shape conv result
        batch_size = inputs.shape[0]
        shaped_result = helper.de_conv_res_shape(conv_res, batch_size, kernel_size, filter_number, zero_padding, input_shape)

        return shaped_result

    def update_weights(self, previous_layer_activation, learning_rate):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']
        back_activation_values = self.cache['back_activation_values']

        # Shape filters and BP activated values for FFT convolution
        shaped_input, shaped_bp_values = helper.dw_compute_shape(previous_layer_activation, back_activation_values, kernel_size, stride, zero_padding, filter_number, input_shape)

        # FFT conv
        conv_res = helper.convolve_filters(shaped_bp_values, shaped_input)

        # Shape conv result
        shaped_result = helper.dw_conv_res_shape(conv_res, back_activation_values, kernel_size, stride, zero_padding, filter_number, input_shape)

        self.filters -= learning_rate * shaped_result

