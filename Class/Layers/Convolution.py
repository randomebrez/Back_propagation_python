import numpy as np
from Class.Layers import LayerBase
from scipy.signal import fftconvolve


# Input & Output shape : (depth, row, col)
class ConvolutionFFTLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True, normalization_function=None):
        self.filters = []
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': kernel_size, 'stride': stride, 'zero_padding': (0, 0), 'use_bias': use_bias}
        self.biases = []
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        self.normalization = normalization_function
        super().__init__('convolution', is_output_layer)

    def initialize(self, input_shape):
        self.parameters['input_shape'] = input_shape
        self.parameters['zero_padding'], self.output_shape = self.padding_and_output_shape_compute()

        kernel_size = self.parameters['kernel_size']
        filter_number = self.parameters['filter_number']
        depth = input_shape[0]

        self.filters = 0.01 * np.random.rand(depth * filter_number, kernel_size, kernel_size)

        if self.parameters['use_bias']:
            self.biases = np.random.rand(self.parameters['filter_number'], 1, 1)

        self.init_cache()

    def init_cache(self):
        self.cache['sigma_primes'] = []
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        batch_size, shaped_inputs = self.shape_input_batch(inputs)  # (3, 3295, 28)
        batch_feature_maps = self.convolve_filters(shaped_inputs, batch_size)

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(batch_feature_maps)
            self.cache['sigma_primes'] = sigma_primes
        else:
            activations = self.activation_function(batch_feature_maps)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def shape_input_batch(self, input_batch):
        kernel_size = self.parameters['kernel_size']
        (batch_size, depth, x_size, y_size) = input_batch.shape
        index_shift = (x_size + kernel_size)
        shaped_input = np.zeros((depth, batch_size * index_shift - kernel_size, y_size))

        for i in range(batch_size):
            index_start = i * index_shift  # 0, 33, 61
            shaped_input[:, index_start:index_start + x_size, :] = input_batch[i]  # 0-->28 |

        return batch_size, shaped_input

    def convolve_filters(self, shaped_input_batch, batch_size):
        stride = self.parameters['stride']
        kernel_size = self.parameters['kernel_size']
        (padding_x, padding_y) = self.parameters['zero_padding']  # (1, 1)
        (depth, input_shape_x, input_shape_y) = self.parameters['input_shape']
        (conv_res_shape_x, conv_res_shape_y) = (kernel_size + input_shape_x - 1, kernel_size + input_shape_y - 1)  # (32, 32)

        # Convolve
        conv_res = fftconvolve(self.filters, shaped_input_batch)  # (3, 3299, 32)

        # Select only the depth slices with full overlap (for each filter)
        full_overlap_conv_res = conv_res[depth-1::depth, :, :]  # (3, 3299, 32)

        # Apply bias
        if self.parameters['use_bias']:
            full_overlap_conv_res += self.biases

        # Indices for truncating single images according to padding
        x_min = (kernel_size - 1) - padding_x  # 3
        x_max = input_shape_x + padding_x        # 29
        y_min = (kernel_size - 1) - padding_y
        y_max = input_shape_y + padding_y

        results = np.empty((batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        index_min = 0
        for i in range(batch_size):
            # Split conv_res into individual convolution result (filters * single_image)
            single_image_conv_res = full_overlap_conv_res[:, index_min:index_min + conv_res_shape_x, :]  # (3, 32, 32) |
            index_min += conv_res_shape_x + 1

            # Truncate single image according to padding
            truncated_conv_res = single_image_conv_res[:, x_min:x_max, y_min:y_max]  # (27, 27) | (26, 26)

            # Slice single image according to stride
            sliced_conv_res = truncated_conv_res[:, ::stride, ::stride]  # (13, 13)

            # Apply bias and save
            results[i] = sliced_conv_res

        return results

    def padding_and_output_shape_compute(self):
        input_shape = self.parameters['input_shape']
        kernel_size = self.parameters['kernel_size']
        stride = self.parameters['stride']

        input_shape_x = input_shape[1]
        input_shape_y = input_shape[2]

        x_jump = (input_shape_x - kernel_size + 1) // stride
        x_padding_needed = kernel_size + stride * x_jump - input_shape_x
        x_padding_each_side = (x_padding_needed + 1) // 2

        if input_shape_x != input_shape_y:
            y_jump = (input_shape_y - kernel_size + 1) // stride
            y_padding_needed = kernel_size + stride * y_jump - input_shape_y
            y_padding_each_side = (y_padding_needed + 1) // 2

            output_shape = (self.parameters['filter_number'], x_jump + 1, y_jump + 1)

            return (x_padding_each_side, y_padding_each_side), output_shape
        else:
            output_shape = (self.parameters['filter_number'], x_jump + 1, x_jump + 1)

            return (x_padding_each_side, x_padding_each_side), output_shape
