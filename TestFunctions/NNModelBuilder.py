import numpy as np
import Class.NetworkBuilder as builder


def perceptron(input_shape, output_shape, hidden_layer_sizes):
    network_builder = builder.NetworkBuilder(input_shape, output_shape)
    # Hidden layers
    for layer_size in hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    # Output layer bloc (dense + OneToOne softmax)
    network_builder.add_dense_layer(np.product(np.asarray(output_shape)))
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)

    network = network_builder.build()

    return network

def auto_encoder(input_shape, output_shape, hidden_layer_sizes, latent_space):
    # Build network
    network_builder = builder.NetworkBuilder(input_shape, output_shape)
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

def convolution(input_shape, output_shape):
    network_builder = builder.NetworkBuilder(input_shape, output_shape)

    network_builder.add_conv_fft_layer(3, 6, 2, 'relu')
    network_builder.add_conv_fft_layer(3, 2, 2, 'relu')
    network_builder.add_flat_layer()

    network_builder.add_dense_layer(800)
    network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(10)
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)
    network = network_builder.build()

    return network