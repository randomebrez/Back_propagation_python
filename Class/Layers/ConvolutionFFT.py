import numpy as np
from Class.Layers import LayerBase
import Tools.ConvolutionHelper as helper
import time

# Input & Output shape : (depth, row, col)
class ConvolutionLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True):
        self.filters = np.array([])
        self.mirror_filters = np.array([])
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': [0, kernel_size, kernel_size], 'stride': [0, stride, stride], 'zero_padding': [0, 0, 0], 'use_bias': use_bias}
        self.use_bias = use_bias
        self.biases = np.array([])
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
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

        initializer = 0.001
        self.filters = np.random.rand(input_depth * filter_number, kernel_size[1], kernel_size[2]) # initializer * (np.random.rand(input_depth * filter_number, kernel_size[1], kernel_size[2]) - 0.5) * 2
        if self.parameters['use_bias']:
            self.biases = initializer * np.random.rand(self.parameters['filter_number'], 1, 1)
            #self.biases = initializer * (np.random.rand(self.output_shape[0], self.output_shape[1], self.output_shape[2]) - 0.5) * 2

        self.init_cache()

    def padding_and_output_shape_compute(self):
        input_shape = self.parameters['input_shape']
        (_, kernel_size_x, kernel_size_y) = self.parameters['kernel_size']
        (_, stride_x, stride_y) = self.parameters['stride']

        input_shape_x = input_shape[1]
        input_shape_y = input_shape[2]

        x_jump = (input_shape_x - kernel_size_x) // stride_x
        x_padding_needed = kernel_size_x + stride_x * x_jump - input_shape_x
        x_padding_each_side = (x_padding_needed + 1) // 2

        if input_shape_x != input_shape_y or kernel_size_x != kernel_size_y or stride_y != stride_x:
            y_jump = (input_shape_y - kernel_size_y) // stride_y
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
        return self.conv_compute(inputs, store)

    def compute_backward_and_update_weights(self, bp_inputs, learning_rate):
        return self.conv_compute_backward(bp_inputs, learning_rate)


    def conv_compute(self, inputs, store):
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
        shaped_feature_maps = helper.input_conv_res_shape(batch_size, batch_feature_maps, kernel_size, stride, zero_padding, input_shape, self.output_shape)

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(shaped_feature_maps)
            self.cache['sigma_primes'] = sigma_primes
        else:
            activations = self.activation_function(shaped_feature_maps)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def conv_compute_backward(self, bp_inputs):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']
        batch_size = bp_inputs.shape[0]

        # compute back_activation values
        back_activation_values = self.cache['sigma_primes'] * bp_inputs
        # self.cache['back_activation_values'] = back_activation_values

        # Shape with stride BP values to use in derivative computations
        shaped_back_activation_values = helper.bp_shape(batch_size, back_activation_values, kernel_size, stride, filter_number, input_shape)
        self.cache['back_activation_values'] = shaped_back_activation_values

        # FFT conv
        conv_res = helper.convolve_filters(self.mirror_filters, shaped_back_activation_values)

        # Shape conv result
        return helper.de_conv_res_shape(batch_size, conv_res, kernel_size, filter_number, zero_padding, input_shape)

    def conv_update(self, previous_layer_activation, learning_rate):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']
        shaped_bp_values = self.cache['back_activation_values']
        tick = time.time()
        batch_size = previous_layer_activation.shape[0]

        # Shape filters for FFT convolution
        shaped_previous_act = helper.dw_compute_shape(batch_size, previous_layer_activation, zero_padding, input_shape)

        # FFT conv
        conv_res = helper.convolve_filters(shaped_bp_values, shaped_previous_act)

        # Shape conv result
        shaped_result = helper.dw_conv_res_shape(batch_size, conv_res, kernel_size, stride, zero_padding, filter_number, input_shape, self.output_shape)

        self.filters -= learning_rate * shaped_result
