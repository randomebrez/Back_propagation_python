import numpy as np
import Manager as manager
import Tools.PlotHelper as ph
import Tools.Computations as computer
import Class.NetworkBuilder as builder
import Class.Model as model


# 'hidden_activation' can be 'sigmoid', 'relu', 'tan_h'
def test_dense_1_hidden(hidden_activation, normalization_function=''):
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
        network_builder.add_dense_layer(layer_size, hidden_activation, normalization_function=normalization_function)

    # Output layer bloc (dense + OneToOne softmax)
    network_builder.add_dense_layer(np.product(np.asarray(output_shape)), 'sigmoid', is_output_layer=True, use_bias=False, normalization_function='')
    network_builder.add_one_to_one_layer('softmax')

    network = network_builder.build()

    # Setup train model
    model_parameters = model.ModelParameters(
        computer.cross_entropy,
        computer.distance_get,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=20,
        epochs=30)

    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_targets, network, model_parameters)
    train_results = manager.train_network(ds_inputs, ds_targets, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_targets, network, model_parameters)

    # Plot results
    ph.plot_perceptron_result(pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test)
    return network


def test_auto_encoder(hidden_activation, normalization_function=''):
    dataset_id = 40996
    feature_name = 'class'
    class_number = 10
    normalization_constant = 255
    batch_size = 100
    # Get dataset from openML
    ds_inputs, ds_targets, input_shape, output_shape = manager.get_column_dataset(dataset_id, feature_name, class_number, normalization_constant, batch_size=batch_size)

    # Choose hidden layer sizes
    hidden_layer_sizes = [700, 100, 700]

    # Build network
    network_builder = builder.NetworkBuilder(input_shape, output_shape)
    for layer_size in hidden_layer_sizes:
        # network_builder.add_dense_layer(layer_size, 'tan_h')
        network_builder.add_dense_layer(layer_size, hidden_activation, normalization_function=normalization_function)

    network_builder.add_dense_layer(np.product(np.asarray(input_shape)), 'sigmoid', is_output_layer=True, use_bias=False,
                                    normalization_function='')

    network = network_builder.build()

    # Setup train model
    model_parameters = model.ModelParameters(
        computer.mean_square_error,
        computer.distance_get,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=20,
        epochs=20)

    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_inputs, network, model_parameters)
    train_results = manager.train_network(ds_inputs, ds_inputs, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_inputs, network, model_parameters)

    # Plot results
    ph.plot_auto_encoder_results(network, ds_inputs[0], pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test, 25)
    return network
