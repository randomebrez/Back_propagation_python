import numpy as np
import Class.NetworkBuilder as builder


def perceptron(input_shape, output_shape, hidden_layer_sizes):
    network_builder = builder.NetworkBuilder(input_shape)
    # Hidden layers
    for layer_size in hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    # Output layer bloc (dense + OneToOne softmax)
    network_builder.add_dense_layer(np.product(np.asarray(output_shape)))
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)

    network = network_builder.build()

    return network

def auto_encoder(input_shape, hidden_layer_sizes, latent_space):
    # Build network
    network_builder = builder.NetworkBuilder(input_shape)
    # encoder block
    for layer_size in hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    # latent space
    network_builder.add_dense_layer(latent_space)
    network_builder.add_one_to_one_layer('relu')

    # decoder block
    for layer_size in reversed(hidden_layer_sizes):
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    # output block (match input shape)
    network_builder.add_dense_layer(np.product(np.asarray(input_shape)))
    network_builder.add_one_to_one_layer('sigmoid', is_output_layer=True)

    network = network_builder.build()

    return network

def convolution(input_shape):
    network_builder = builder.NetworkBuilder(input_shape)

    network_builder.add_convolution_layer(3, 6, 2, 'relu')  # (3, 12, 12)
    network_builder.add_pool_layer(2, mode='max')  # (3, 6, 6)
    network_builder.add_convolution_layer(6, 2, 2, 'relu')  # (6, 3, 3)
    network_builder.add_flat_layer()

    network_builder.add_dense_layer(3 * 3 * 6)
    network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(10)
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)
    network = network_builder.build()

    return network

def ae_semi_conv(input_shape, hidden_layer_sizes, latent_space):
    network_builder = builder.NetworkBuilder(input_shape)  # (1, 28, 28)

    # Convolutional encoder
    network_builder.add_convolution_layer(3, 6, 2, 'relu')  # (3, 12, 12)
    network_builder.add_pool_layer(2, mode='max')  # (3, 6, 6)
    network_builder.add_convolution_layer(6, 2, 2, 'relu')  # (6, 3, 3)
    network_builder.add_flat_layer()

    network_builder.add_dense_layer(latent_space)
    network_builder.add_one_to_one_layer('relu')

    # decoder block
    for layer_size in reversed(hidden_layer_sizes):
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    # output block (match input shape)
    network_builder.add_dense_layer(np.product(np.asarray(input_shape)))
    network_builder.add_one_to_one_layer('sigmoid')
    network_builder.add_reshape_layer(input_shape, is_output_layer=True)
    network = network_builder.build()

    return network


def ae_convolution(input_shape, latent_space):
    network_builder = builder.NetworkBuilder(input_shape)  # (1, 28, 28)

    network_builder.add_convolution_layer(3, 6, 2, 'relu')  # (3, 12, 12)
    network_builder.add_pool_layer(2, mode='max')  # (3, 6, 6)
    network_builder.add_convolution_layer(6, 2, 2, 'relu')  # (6, 3, 3)
    network_builder.add_flat_layer()

    network_builder.add_dense_layer(latent_space)
    network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(6*3*3)
    network_builder.add_one_to_one_layer('relu')

    network_builder.add_reshape_layer((6, 3, 3))  # (6, 3, 3)
    network_builder.add_transposed_conv_layer(6, 3, 2, 1, 'relu')  # (3, 5, 5)
    network_builder.add_transposed_conv_layer(3, 5, 2, 1, 'relu')  # (3, 11, 11)
    network_builder.add_transposed_conv_layer(1, 8, 2, 0, 'sigmoid', is_output_layer=True)  # (1, 28, 28)
    network = network_builder.build()

    return network