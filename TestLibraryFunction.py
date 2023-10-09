import numpy as np
import time
import Manager as manager
import Tools.PlotHelper as ph
import Tools.Computations as computer
import Class.NetworkBuilder as builder
import Class.Model as model


# 'hidden_activation' can be 'sigmoid', 'relu', 'tan_h'
def test_dense_1_hidden():
    # Get dataset from openML
    dataset_id = 40996
    feature_name = 'class'
    class_number = 10
    normalization_constant = 255
    batch_size = 100
    ds_inputs, ds_targets, input_shape, output_shape = manager.get_column_dataset(dataset_id, feature_name, class_number, normalization_constant, batch_size=batch_size)

    # Choose hidden layer sizes
    hidden_layer_sizes = [700]

    network_builder = builder.NetworkBuilder(input_shape, output_shape)
    for layer_size in hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    # Output layer bloc (dense + OneToOne softmax)
    network_builder.add_dense_layer(np.product(np.asarray(output_shape)))
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)

    network = network_builder.build()

    # Setup train model
    model_parameters = model.ModelParameters(
        computer.cross_entropy,
        computer.cross_entropy_derivative,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=5,
        epochs=30)

    start = time.time()

    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_targets, network, model_parameters)
    train_results = manager.train_network(ds_inputs, ds_targets, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_targets, network, model_parameters)

    # Plot results
    ph.plot_perceptron_result(pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test)

    tick = time.time()
    print("FFT network execution time : {0}".format(tick - start))
    return network


def test_auto_encoder():
    dataset_id = 40996
    feature_name = 'class'
    class_number = 10
    normalization_constant = 255
    batch_size = 100
    # Get dataset from openML
    ds_inputs, ds_targets, input_shape, output_shape = manager.get_column_dataset(dataset_id, feature_name, class_number, normalization_constant, batch_size=batch_size)

    # Choose hidden layer sizes
    hidden_layer_sizes = [700, 100]

    # Build network
    network_builder = builder.NetworkBuilder(input_shape, output_shape)
    for layer_size in hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    for layer_size in reversed(hidden_layer_sizes[:-1]):
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(np.product(np.asarray(input_shape)))
    network_builder.add_one_to_one_layer('sigmoid', is_output_layer=True)

    network = network_builder.build()

    # Setup train model
    model_parameters = model.ModelParameters(
        computer.mean_square_error,
        computer.distance_get,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=5,
        epochs=20)

    start = time.time()

    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_inputs, network, model_parameters, with_details=False)
    train_results = manager.train_network(ds_inputs, ds_inputs, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_inputs, network, model_parameters, with_details=False)

    # Plot results
    ph.plot_auto_encoder_results(network, ds_inputs[0], pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test, 36)

    tick = time.time()
    print("FFT network execution time : {0}".format(tick - start))
    return network

def test_conv_net():
    dataset_id = 40996
    feature_name = 'class'
    class_number = 10
    normalization_constant = 255
    batch_size = 100
    ds_inputs, ds_targets, input_shape, output_shape = manager.get_image_dataset(dataset_id, feature_name, class_number, normalization_constant, batch_size=batch_size)

    network_builder = builder.NetworkBuilder(input_shape, output_shape)

    network_builder.add_conv_fft_layer(3, 6, 2, 'relu')
    # network_builder.add_pool_layer(2, 'max')
    network_builder.add_flat_layer()

    network_builder.add_dense_layer(800)
    network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(10)
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)
    network = network_builder.build()

    train_model = model.ModelParameters(
        computer.cross_entropy,
        computer.cross_entropy_derivative,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=5,
        epochs=10)

    start = time.time()

    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_targets, network, train_model)
    train_results = manager.train_network(ds_inputs, ds_targets, network, train_model)
    post_train_test = manager.test_network(ds_inputs, ds_targets, network, train_model)

    # Plot results
    ph.plot_perceptron_result(pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test)

    tick = time.time()
    print("FFT network execution time : {0}".format(tick - start))
    return network
