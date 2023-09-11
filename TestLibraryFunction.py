import numpy as np
import Manager as manager
import Tools.Computations as computer
import Class.NetworkBuilder as builder
import Class.Model as model


# 'hidden_activation' can be 'sigmoid', 'relu', 'tan_h'
def test_dense_1_hidden(hidden_activation, normalization_function=''):
    # Get dataset from openML
    dataset_id = 40996
    feature_name = 'class',
    normalization_constant = 255
    batch_size = 100
    ds_inputs, ds_targets = manager.get_dataset(dataset_id, feature_name, normalization_constant, batch_size=batch_size)

    input_size = np.shape(ds_inputs[0][0])[0]
    output_size = np.shape(ds_targets[0][0])[0]
    # Choose hidden layer sizes
    hidden_layer_sizes = [700]

    network_builder = builder.NetworkBuilder(input_size, output_size)
    for layer_size in hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size, hidden_activation, normalization_function=normalization_function)

    # Output layer bloc (dense + OneToOne softmax)
    network_builder.add_dense_layer(output_size, 'sigmoid', is_output_layer=True, use_bias=False, normalization_function='')
    network_builder.add_one_to_one_layer(output_size, 'softmax')

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
    results = manager.train_network(ds_inputs, ds_targets, network, model_parameters)

    # Plot results
    manager.plot_result(results['pre_train_test'], results['batch_costs'], results['mean_batch_costs'], results['post_train_test'])
