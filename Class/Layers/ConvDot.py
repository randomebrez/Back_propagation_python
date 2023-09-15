import numpy as np
from Class.Layers import LayerBase


# Input & Output shape : (depth, row, col)
class ConvolutionDotLayer(LayerBase.__LayerBase):
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

        row_length = kernel_size * kernel_size * depth
        self.filters = 0.01 * np.random.rand(filter_number, row_length)

        if self.parameters['use_bias']:
            self.biases = np.random.rand(self.parameters['filter_number'], 1)

        self.init_cache()

    def init_cache(self):
        self.cache['sigma_primes'] = []
        self.cache['back_activation_values'] = []

    def clean_cache(self):
        super().clean_cache()
        self.init_cache()

    def compute(self, inputs, store):
        batch_image_col = self.im_to_col(inputs)
        aggregation_result = np.dot(self.filters, batch_image_col) + self.biases

        # If backpropagation is needed, store d_activation values
        if store:
            activations, sigma_primes = self.activation_function_with_derivative(aggregation_result)
            self.cache['sigma_primes'] = self.col_to_im(sigma_primes)
        else:
            activations = self.activation_function(aggregation_result)

        # Reshape to output a 4D vector (batch_size, depth, row, col)
        output_shaped_activations = self.im_to_col(activations)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = output_shaped_activations

        return output_shaped_activations

    def padding_and_output_shape_compute(self):
        input_shape = self.parameters['input_shape']
        kernel_size = self.parameters['kernel_size']
        stride = self.parameters['stride']

        input_shape_x = input_shape[1]
        input_shape_y = input_shape[2]

        x_jump = int((input_shape_x - kernel_size) / stride) + 1
        x_padding_needed = kernel_size + stride * x_jump - input_shape_x
        x_padding_each_side = int((x_padding_needed + 1) / 2)

        if input_shape_x != input_shape_y:
            y_jump = int((input_shape_y - kernel_size) / stride) + 1
            y_padding_needed = kernel_size + stride * y_jump - input_shape_y
            y_padding_each_side = int((y_padding_needed + 1) / 2)

            output_shape = (self.parameters['filter_number'], x_jump + 1, y_jump + 1)

            return (x_padding_each_side, y_padding_each_side), output_shape
        else:
            output_shape = (self.parameters['filter_number'], x_jump + 1, x_jump + 1)

            return (x_padding_each_side, x_padding_each_side), output_shape

    # 'image batch' : 4D vector (batch_size, depth, x, y)
    def im_to_col(self, image_batch):
        depth, input_shape_x, input_shape_y = self.parameters['input_shape']
        kernel_size = self.parameters['kernel_size']
        zero_padding_x, zero_padding_y = self.parameters['zero_padding']
        stride = self.parameters['stride']
        batch_size = image_batch.shape[0]

        nb_block_x = (input_shape_x + 2 * zero_padding_x - kernel_size) // stride + 1
        nb_block_y = (input_shape_y + 2 * zero_padding_y - kernel_size) // stride + 1

        block_to_col_size = kernel_size * kernel_size * depth
        image_col_number = nb_block_x * nb_block_y

        batch_padded = np.zeros((batch_size, depth, input_shape_x + 2 * zero_padding_x, input_shape_y + 2 * zero_padding_y))
        batch_padded[:, :, zero_padding_x:-zero_padding_x, zero_padding_y:-zero_padding_y] = image_batch

        result = np.empty((block_to_col_size, batch_size * image_col_number))
        for x in range(nb_block_x):
            x_min_index = x * stride
            for y in range(nb_block_y):
                y_min_index = y * stride
                sub_images = batch_padded[:, :, x_min_index:x_min_index + kernel_size, y_min_index:y_min_index + kernel_size]
                for batch_image_index in range(batch_size):
                    sub_image_padded = sub_images[batch_image_index, :, :, :]
                    column_index = batch_image_index * image_col_number + y + nb_block_y * x
                    result[:, column_index] = sub_image_padded.reshape((block_to_col_size, 1))
        return result

    def col_to_im(self, image_batch):
        depth, input_shape_x, input_shape_y = self.parameters['input_shape']
        kernel_size = self.parameters['kernel_size']
        zero_padding_x, zero_padding_y = self.parameters['zero_padding']
        stride = self.parameters['stride']

        nb_block_x = (input_shape_x + 2 * zero_padding_x - kernel_size) // stride + 1
        nb_block_y = (input_shape_y + 2 * zero_padding_y - kernel_size) // stride + 1

        image_col_number = nb_block_x * nb_block_y
        batch_size = image_batch.shape[1] / image_col_number

        result = np.empty((batch_size, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
        for i in range(batch_size):
            y_min_index = i * image_col_number
            single_image = image_batch[:, y_min_index:y_min_index + image_col_number]
            result[i, :, :, :] = single_image.reshape((self.output_shape[0], self.output_shape[1], self.output_shape[2]))

        return result
