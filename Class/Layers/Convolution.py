import numpy as np
from Class.Layers import LayerBase

# Input & Output shape : (depth, row, col)
class ConvolutionLayer(LayerBase.__LayerBase):
    def __init__(self, filter_number, kernel_size, stride, activation_function, activation_function_with_derivative, is_output_layer=False, use_bias=True):
        self.filters = np.array([])
        self.biases = np.array([])
        self.parameters = {'input_shape': (0, 0, 0), 'filter_number': filter_number, 'kernel_size': [0, kernel_size, kernel_size], 'stride': [stride, stride], 'zero_padding': [0, 0], 'use_bias': use_bias}
        self.use_bias = use_bias
        self.activation_function = activation_function
        self.activation_function_with_derivative = activation_function_with_derivative
        super().__init__('convolution', is_output_layer)

    def initialize(self, input_shape):
        input_depth = input_shape[0]
        self.parameters['input_shape'] = input_shape
        self.parameters['zero_padding'], self.output_shape = self.padding_and_output_shape_compute()

        kernel_size = self.parameters['kernel_size']
        filter_number = self.parameters['filter_number']

        kernel_size[0] = input_depth

        initializer = 0.001
        self.filters = np.random.rand(input_depth * filter_number, kernel_size[1], kernel_size[2]) # initializer * (np.random.rand(input_depth * filter_number, kernel_size[1], kernel_size[2]) - 0.5) * 2
        if self.parameters['use_bias']:
            #self.biases = 2 * np.random.rand(self.parameters['filter_number'], 1, 1) - 0.5)
            self.biases = np.zeros((self.parameters['filter_number'], 1, 1))

        self.init_cache()

    def padding_and_output_shape_compute(self):
        input_shape = self.parameters['input_shape']
        (_, kernel_size_x, kernel_size_y) = self.parameters['kernel_size']
        (stride_x, stride_y) = self.parameters['stride']

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
                f = self.filters[f_min:f_min + input_shape[0]]
                product = patch * f
                conv_result = np.sum(product, axis=(1, 2, 3)).reshape((batch_size, 1))
                feature_maps[:, f_min:f_min + input_shape[0], x, y] += conv_result

        # Apply bias
        if self.parameters['use_bias']:
            feature_maps += self.biases

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(feature_maps)
            self.cache['sigma_primes'] = sigma_primes
            self.cache['inputs'] = inputs
        else:
            activations = self.activation_function(feature_maps)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    def compute_backward_and_update_weights(self, inputs, learning_rate):
        input_shape = self.parameters['input_shape']
        filter_number = self.parameters['filter_number']
        kernel_size = self.parameters['kernel_size']
        zero_padding = self.parameters['zero_padding']
        stride = self.parameters['stride']

        sigma_primes = self.cache['sigma_primes']
        previous_layer_activation = self.cache['inputs']

        back_activation_values = sigma_primes * inputs

        batch_size = back_activation_values.shape[0]
        bp_outputs = np.zeros((batch_size,) + input_shape)
        dE_dw = np.zeros(self.filters.shape)

        for filter_index in range(filter_number):
            for patch, x, y in self.patches_generator(bp_outputs, input_shape, kernel_size, stride, zero_padding):
                f_min = filter_index * input_shape[0]
                temp = back_activation_values[:, filter_index, x, y]
                patch += self.filters[f_min:f_min + input_shape[0]] * temp[:, np.newaxis, np.newaxis, np.newaxis]

            for patch, x, y in self.patches_generator(previous_layer_activation, input_shape, kernel_size, stride, zero_padding):
                f_min = filter_index * input_shape[0]
                dE_dw[f_min:f_min + input_shape[0]] += np.sum(patch * back_activation_values[:, filter_index, x, y].reshape((batch_size, 1, 1, 1)), axis=0)

        self.filters -= learning_rate * dE_dw / batch_size

        if self.use_bias:
            bias_variation = learning_rate * np.mean(back_activation_values, axis=(0, 2, 3), keepdims=True).reshape((filter_number, 1, 1))
            self.biases -= bias_variation
        return bp_outputs


    def patches_generator(self, inputs, input_shape, kernel_size, stride, padding):
        input_x, input_y = input_shape[1], input_shape[2]
        kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
        padding_x, padding_y = padding[0], padding[1]
        stride_x, stride_y = stride[0], stride[1]
        for x in range((input_x - kernel_size_x + 2 * padding_x) // stride_x + 1):
            x_min = x * stride_x
            x_max = x_min + kernel_size_x
            for y in range((input_y - kernel_size_y + 2 * padding_y) // stride_y + 1):
                y_min = y * stride_y
                y_max = y_min + kernel_size_y
                patch = inputs[:, :, x_min:x_max, y_min:y_max]
                yield patch, x, y
