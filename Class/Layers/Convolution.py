import numpy as np
from Class.Layers import LayerBase
import Tools.ConvolutionHelper as helper
import time

# Input & Output shape : (depth, row, col)
class ConvolutionFFTLayer(LayerBase.__LayerBase):
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

        self.filters = 0.01 * (np.random.rand(input_depth * filter_number, kernel_size[1], kernel_size[2]) - 0.5) * 2
        if self.parameters['use_bias']:
            self.biases = 0.01 * np.random.rand(self.parameters['filter_number'], 1, 1)

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
        #return self.conv_compute(inputs, store)
        return self.patch_compute(inputs, store)

    def compute_backward(self, inputs):
        # return self.conv_compute_backward(inputs)
        return self.patch_compute_backward(inputs)

    def update_weights(self, previous_layer_activation, learning_rate):
        #self.conv_update(previous_layer_activation, learning_rate)
        self.patch_update(previous_layer_activation, learning_rate)


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

    def patch_compute(self, inputs, store):
        kernel_size = self.parameters['kernel_size']
        stride = self.parameters['stride']
        zero_padding = self.parameters['zero_padding']
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        batch_size = inputs.shape[0]

        feature_maps = np.zeros((batch_size, filter_number, self.output_shape[1], self.output_shape[2]))
        for patch, x, y in self.patches_generator(inputs, input_shape, kernel_size, stride, zero_padding):
            for filter_index in range(filter_number):
                f_min = filter_index * input_shape[0]
                f = self.filters[f_min:f_min+input_shape[0]]
                product = patch * f
                conv_result = np.sum(product, axis=(1,2,3)).reshape(batch_size, 1)
                feature_maps[:, f_min:f_min+input_shape[0], x, y] += conv_result

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(feature_maps)
            self.cache['sigma_primes'] = sigma_primes
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

    def conv_compute_backward(self, inputs):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']
        batch_size = inputs.shape[0]

        # compute back_activation values
        back_activation_values = self.cache['sigma_primes'] * inputs
        # self.cache['back_activation_values'] = back_activation_values

        # Shape with stride BP values to use in derivative computations
        shaped_back_activation_values = helper.bp_shape(batch_size, back_activation_values, kernel_size, stride, filter_number, input_shape)
        self.cache['back_activation_values'] = shaped_back_activation_values

        # FFT conv
        conv_res = helper.convolve_filters(self.mirror_filters, shaped_back_activation_values)

        # Shape conv result
        return helper.de_conv_res_shape(batch_size, conv_res, kernel_size, filter_number, zero_padding, input_shape)

    def patch_compute_backward(self, inputs):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']
        sigma_primes = self.cache['sigma_primes']

        back_activation_values = sigma_primes * inputs
        self.cache['back_activation_values'] = back_activation_values

        batch_size = back_activation_values.shape[0]
        bp_outputs = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        for patch, x, y in self.patches_generator(bp_outputs, input_shape, kernel_size, stride, zero_padding):
            for filter_index in range(filter_number):
                f_min = filter_index * input_shape[0]
                temp = back_activation_values[:, filter_index, x, y]
                patch += self.filters[f_min:f_min+input_shape[0]] * temp[:, np.newaxis, np.newaxis, np.newaxis]
        return bp_outputs

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

    def patch_update(self, previous_layer_activation, learning_rate):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']
        shaped_bp_values = self.cache['back_activation_values']
        batch_size = previous_layer_activation.shape[0]

        dE_dw = np.zeros(self.filters.shape)
        for patch, x, y in self.patches_generator(previous_layer_activation, input_shape, kernel_size, stride, zero_padding):
            for filter_index in range(filter_number):
                f_min = filter_index * input_shape[0]
                dE_dw[f_min:f_min+input_shape[0]] += np.sum(patch * shaped_bp_values[:, filter_index, x, y].reshape(batch_size, 1, 1, 1), axis=0)

        self.filters -= learning_rate * dE_dw / batch_size

        if self.use_bias:
            bias_variation = learning_rate * np.mean(shaped_bp_values, axis=(0,2,3), keepdims=True).reshape((filter_number, 1, 1))
            self.biases -= bias_variation

    def patches_generator(self, inputs, input_shape, kernel_size, stride, padding):
        input_x, input_y = input_shape[1], input_shape[2]
        padding_x, padding_y = padding[1], padding[2]
        kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
        stride_x, stride_y = stride[1], stride[2]
        for x in range((input_x - kernel_size_x + 2 * padding_x) // stride_x + 1):
            x_min = x * stride_x
            x_max = x_min + kernel_size_x
            for y in range((input_y - kernel_size_y + 2 * padding_y) // stride_y + 1):
                y_min = y * stride_y
                y_max = y_min + kernel_size_y
                patch = inputs[:, :, x_min:x_max, y_min:y_max]
                yield patch, x, y
