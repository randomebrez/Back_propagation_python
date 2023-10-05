import numpy as np
from Class.Layers import LayerBase
from scipy.signal import fftconvolve


# Input & Output shape : (depth, row, col)
class ConvolutionFFTLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True, normalization_function=None):
        self.filters = []
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': (0, kernel_size, kernel_size), 'stride': (0, stride, stride), 'zero_padding': (0, 0, 0), 'use_bias': use_bias}
        self.biases = []
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        self.normalization = normalization_function
        super().__init__('convolution', is_output_layer)

    def initialize(self, input_shape):
        input_depth = input_shape[0]
        self.parameters['input_shape'] = input_shape
        self.parameters['zero_padding', 1:], self.output_shape = self.padding_and_output_shape_compute()

        # Set depth wise parameters already known by input shape
        self.parameters['kernel_size', 0] = input_depth
        self.parameters['zero_padding', 0] = input_depth

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
        shaped_inputs = self.input_compute_shape(inputs, kernel_size[1])

        # Convolve
        batch_feature_maps = self.convolve_filters(shaped_inputs, self.filters)

        # Shape convolution result according to stride & padding
        shaped_feature_maps = self.input_conv_res_shape(batch_feature_maps, batch_size, kernel_size, stride, zero_padding, input_shape, self.output_shape)

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
        filter_mirror, shaped_bp_values = self.de_compute_shape(self.filters, back_activation_values, filter_number, kernel_size, stride, input_shape)

        # FFT conv
        conv_res = self.convolve_filters(filter_mirror, shaped_bp_values)

        # Shape conv result
        batch_size = inputs.shape[0]
        shaped_result = self.de_conv_res_shape(conv_res, batch_size, kernel_size, filter_number, zero_padding, input_shape)

        return shaped_result

    def update_weights(self, previous_layer_activation, learning_rate):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        back_activation_values = self.cache['back_activation_values']

        # Shape filters and BP activated values for FFT convolution
        shaped_input, shaped_bp_values = self.dw_compute_shape(previous_layer_activation, back_activation_values, filter_number, input_shape)

        # FFT conv
        conv_res = self.convolve_filters(shaped_bp_values, shaped_input)

        # Shape conv result
        shaped_result = self.de_conv_res_shape(conv_res, back_activation_values, kernel_size, filter_number, zero_padding, input_shape)

        self.filters -= learning_rate * shaped_result

    def input_compute_shape(self, input_batch, kernel_size_x):
        (batch_size, depth, x_size, y_size) = input_batch.shape
        index_shift = (x_size + kernel_size_x)
        shaped_input = np.zeros((depth, batch_size * index_shift - kernel_size_x, y_size))

        for i in range(batch_size):
            index_start = i * index_shift
            shaped_input[:, index_start:index_start + x_size] = input_batch[i]

        return shaped_input

    def input_conv_res_shape(self, convolution_result, batch_size, kernel_size, stride, padding, input_shape, output_shape):
        padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
        kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
        stride_x, stride_y = stride[0], stride[1]
        input_depth, input_shape_x, input_shape_y = input_shape[0], input_shape[1], input_shape[2]

        # Truncate & Slice according to z with only full overlap (for each filter)
        full_overlap_conv_res = convolution_result[input_depth - 1::input_depth]

        # Split above result into batch_size array of single outputs
        conv_res_split = np.split(full_overlap_conv_res, batch_size, axis=0)

        # Indices for truncating single images according to padding
        x_to_remove_each_side = (kernel_size_x - 1) - padding_x
        y_to_remove_each_side = (kernel_size_y - 1) - padding_y

        results = np.empty((batch_size, output_shape[0], output_shape[1], output_shape[2]))
        for i in range(batch_size):
            single_image = conv_res_split[i]
            # Remove 'marker' column of the convolution result (except for last one)
            if i != batch_size - 1:
                single_image = single_image[:, -1]

            # Truncate image according to padding
            single_image = single_image[:, x_to_remove_each_side:-x_to_remove_each_side, y_to_remove_each_side::-y_to_remove_each_side]

            # Slice single image according to stride
            sliced_conv_res = single_image[:, ::stride_x, ::stride_y]

            # Save
            results[i] = sliced_conv_res
        return results

    def dw_compute_shape(self, previous_layer_activation, back_prop_activated_values, filter_number, input_shape):
        input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
        batch_size, bp_x, bp_y = back_prop_activated_values.shape[0], back_prop_activated_values.shape[2], back_prop_activated_values.shape[3]

        # size of 1 bloc : InterestingValues | kernel_size 0
        x_shift = (bp_x + input_x)

        bp_shape_depth = (filter_number - 1) * (input_depth - 1) + filter_number
        bp_shape_x = x_shift * batch_size - input_x
        shaped_bp_values = np.zeros((bp_shape_depth, bp_shape_x, bp_y))

        input_shape_x = x_shift * batch_size - bp_x
        shaped_input = np.zeros((input_depth, input_shape_x, input_y))

        for i in range(batch_size):
            index_start = i * x_shift
            shaped_input[:, index_start:index_start + input_x] = previous_layer_activation[i]
            shaped_bp_values[::input_depth, index_start:index_start + bp_x] = back_prop_activated_values[i]

        return shaped_input, shaped_bp_values

    def dw_conv_res_shape(self, convolution_result, back_prop_activated_values, kernel_size, filter_number, padding, input_shape):
        input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
        padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
        batch_size, bp_x, bp_y = back_prop_activated_values.shape[0], back_prop_activated_values.shape[2], back_prop_activated_values.shape[3]

        # Truncate depth wise (only overlap of feature map with full input depth)
        depth_split = np.split(convolution_result, input_depth + 1)

        # Indices for truncating single images according to padding
        x_to_remove_each_side = (batch_size - 1) * (bp_x + input_x) + input_x - 1 - padding_x
        y_to_remove_each_side = (bp_y - 1) - padding_y

        filter_results = np.zeros((input_depth * filter_number, kernel_size[1], kernel_size[2]))
        for i in range(filter_number):

            filter_result = depth_split[i]
            if i != filter_number - 1:
                filter_result = filter_result[:-1]

            # Remove useless values row/col wise
            filter_result = filter_result[:, x_to_remove_each_side:-x_to_remove_each_side, y_to_remove_each_side:-y_to_remove_each_side]

            # Mean on each input of the batch
            filter_results[i * input_depth:(i+1) * input_depth] = filter_result / batch_size

        return filter_results

    def de_compute_shape(self, filters, back_prop_activated_values, filter_number, kernel_size, stride, input_shape):
        stride_x, stride_y = stride[0], stride[1]
        input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
        batch_size, bp_x, bp_y = back_prop_activated_values.shape[0], back_prop_activated_values.shape[2], back_prop_activated_values.shape[3]

        # Mirror filter block
        filter_mirror = np.flip(np.flip(np.flip(filters, 0), 1), 2)

        # size of 1 bloc : InterestingValues | kernel_size 0
        x_shift = bp_x + (bp_x - 1) * (stride_x - 1) + kernel_size[1]

        bp_shape_depth = (filter_number - 1) * (input_depth - 1) + filter_number
        bp_shape_x = x_shift * batch_size - kernel_size[1]
        bp_shape_y = bp_y + (bp_y - 1) * (stride_y - 1)
        shaped_bp_values = np.zeros((bp_shape_depth, bp_shape_x, bp_shape_y))

        for i in range(batch_size):
            index_start = i * x_shift
            shaped_bp_values[::input_depth, index_start:index_start + bp_x:stride_x, ::stride_y] = np.flip(back_prop_activated_values[i], 0)

        return filter_mirror, shaped_bp_values

    def de_conv_res_shape(self, convolution_result, batch_size, kernel_size, filter_number, padding, input_shape):
        input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
        padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
        kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]

        # Truncate depth wise (only overlap of feature map with full input depth)
        z_to_remove = (filter_number - 1) * input_depth
        depth_truncate = convolution_result[z_to_remove:-z_to_remove]

        single_split = np.split(depth_truncate, batch_size, axis=1)
        mean_values = np.sum(single_split[:-1], axis=0)

        # last one don't have a marker column to remove
        mean_values += single_split[-1]

        # Indices for truncating single images according to padding
        x_to_remove_each_side = kernel_size_x - padding_x - 1
        y_to_remove_each_side = kernel_size_y - padding_y - 1
        truncated_mean_values = mean_values[:, x_to_remove_each_side:-x_to_remove_each_side, y_to_remove_each_side:-y_to_remove_each_side]

        return truncated_mean_values / (batch_size * filter_number)

    # Convolve using FFT
    def convolve_filters(self, kernel, shaped_input_batch):
        conv_res = fftconvolve(kernel, shaped_input_batch)
        return conv_res