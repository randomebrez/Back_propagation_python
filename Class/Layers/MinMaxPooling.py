import numpy as np
from Class.Layers import LayerBase


# Input shape : (depth, row, col) => Output shape : (depth, row, col)
class MinMaxPoolingLayer(LayerBase.__LayerBase):
    epsilon = 0.001

    def __init__(self, kernel_size, mode='max', is_output_layer=False):
        self.parameters = {'input_shape': (0, 0, 0), 'kernel_size': kernel_size, 'mode': mode}
        super().__init__('pool', is_output_layer)

    def initialize(self, input_shape):
        k_size = self.parameters['kernel_size']

        self.parameters['input_shape'] = input_shape
        self.output_shape = (input_shape[0], input_shape[1] // k_size, input_shape[2] // k_size)

    def compute(self, inputs, store):
        row_wise_result = self.dimension_wise_compute(0, inputs, store)
        activations = self.dimension_wise_compute(1, row_wise_result, store)

        # Store activations if asked, or for output layer
        if store or self.is_output_layer:
            self.cache['activation_values'] = activations

        return activations

    # backward inputs : rows = neuron activation - column = batch index
    def compute_backward(self, inputs):
        k_size = self.parameters['kernel_size']
        input_shape = self.parameters['input_shape']
        col_wise_values = self.cache['c_wise']
        row_wise_values = self.cache['r_wise']

        # Dilate column wise
        c_dilate = np.zeros((col_wise_values.shape[1], col_wise_values.shape[2], col_wise_values.shape[3],
                             k_size * col_wise_values.shape[4]))
        for i in range(col_wise_values.shape[0]):
            for j in range(col_wise_values.shape[4]):
                c_dilate[:, :, :, i + j * k_size] = col_wise_values[i][:, :, :, j] * inputs[:, :, :, j]

        # Update previous values in a new variable to not update cache
        # test if reshape (1, c_dilate.shape) puis multiplier le bloc marche ? et est + rapide ?
        updated_values = np.empty(row_wise_values.shape)
        for i in range(row_wise_values.shape[0]):
            updated_values[i] = c_dilate * row_wise_values[i]

        # Dilate row wise
        backward_output = np.zeros((row_wise_values.shape[1], input_shape[0], input_shape[1], input_shape[2]))
        for i in range(row_wise_values.shape[0]):
            for j in range(row_wise_values.shape[3]):
                backward_output[:, :, i + j * k_size] = updated_values[i][:, :, j]

        return backward_output

    def dimension_wise_compute(self, axis: int, initial_matrix, store):
        k_size = self.parameters['kernel_size']
        switch = 1
        if self.parameters['mode'] == 'min':
            switch = -1

        # ['A_0', 'A_prime', ..., 'A_n']
        if axis == 0:
            cache_index = 'r_wise'
            sub_matrices = np.empty((k_size, initial_matrix.shape[0], initial_matrix.shape[1],
                                     initial_matrix.shape[2] // k_size, initial_matrix.shape[3]))
            for i in range(k_size):
                sub_matrices[i] = initial_matrix[:, :, i::k_size]
        else:
            cache_index = 'c_wise'
            sub_matrices = np.empty((k_size, initial_matrix.shape[0], initial_matrix.shape[1], initial_matrix.shape[2],
                                     initial_matrix.shape[3] // k_size))
            for i in range(k_size):
                sub_matrices[i] = initial_matrix[:, :, :, i::k_size]

        max_values = sub_matrices[-1]
        for i in range(k_size - 1):
            challenger = sub_matrices[i]
            max_values = 0.5 * (max_values + challenger + switch * np.abs(max_values - challenger))

        # save for backprop
        if store:
            deltas = np.empty(
                (k_size, max_values.shape[0], max_values.shape[1], max_values.shape[2], max_values.shape[3]))
            for i in range(k_size):
                on_off_m = switch * (max_values - sub_matrices[i])
                deltas[i] = np.ones(on_off_m.shape) - (on_off_m // (np.abs(on_off_m) - self.epsilon))
            self.cache[cache_index] = deltas

        return max_values
