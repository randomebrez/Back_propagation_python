import numpy as np
import time
import Manager as manager
import Tools.PlotHelper as ph
import Tools.Computations as computer
import Class.NetworkBuilder as builder
import Class.Model as model

# Get dataset from openML
def get_dataset():
    dataset_id = 40996
    feature_name = 'class'
    class_number = 10
    normalization_constant = 255
    batch_size = 100

    return manager.get_column_dataset(dataset_id, feature_name, class_number, normalization_constant, batch_size=batch_size)

def test_perceptron():
    ds_inputs, ds_targets, input_shape, output_shape = get_dataset()
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
    ds_inputs, ds_targets, input_shape, output_shape = get_dataset()

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
    print("Perceptron network execution time : {0}".format(tick - start))
    return network

def test_conv_net():
    ds_inputs, ds_targets, input_shape, output_shape = get_dataset()

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
    print("Auto-encoder network execution time : {0}".format(tick - start))
    return network

def test_perceptron_ae_combined():
    ds_inputs, ds_targets, input_shape, output_shape = get_dataset()

    # Perceptron
    network_builder = builder.NetworkBuilder(input_shape, output_shape)
    perceptron_train_model = model.ModelParameters(
        computer.cross_entropy,
        computer.cross_entropy_derivative,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=5,
        epochs=10)

    perceptron_hidden_layer_sizes = [800]
    for layer_size in perceptron_hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(np.product(np.asarray(output_shape)))
    network_builder.add_one_to_one_layer('softmax', is_output_layer=True)

    perceptron = network_builder.build()

    start = time.time()

    manager.train_network(ds_inputs, ds_targets, perceptron, perceptron_train_model)
    perceptron_test_result = manager.test_network(ds_inputs, ds_targets, perceptron, perceptron_train_model)

    tick = time.time()
    print("Perceptron network execution time : {0}".format(tick - start))

    # Auto_encoding
    ae_train_model = model.ModelParameters(
        computer.mean_square_error,
        computer.distance_get,
        initial_learning_rate=0.1,
        final_learning_rate=0.01,
        learning_rate_steps=5,
        epochs=10)
    ae_hidden_layer_sizes = [700, 100, 700]
    network_builder.create_new(input_shape, input_shape)
    for layer_size in ae_hidden_layer_sizes:
        network_builder.add_dense_layer(layer_size)
        network_builder.add_one_to_one_layer('relu')

    network_builder.add_dense_layer(np.product(np.asarray(input_shape)))
    network_builder.add_one_to_one_layer('sigmoid', is_output_layer=True)

    auto_encoding = network_builder.build()

    tick = time.time()

    manager.train_network(ds_inputs, ds_inputs, auto_encoding, ae_train_model)
    ae_test_result = manager.test_network(ds_inputs, ds_inputs, auto_encoding, ae_train_model)
    print("Auto_encoder network execution time : {0}".format(time.time() - tick))

    test_inputs, test_targets = ds_inputs[1], ds_targets[1]
    ae_outputs = np.zeros(test_inputs.shape)
    for i in range(test_inputs.shape[0]):
        # Compute using auto-encoding
        auto_encoding.feed_forward(test_inputs[i], False)
        ae_outputs[i] = auto_encoding.get_outputs()

    tick = time.time()
    # Compute perceptron with ae outputs
    combined_test_result = perceptron.test(ae_outputs, test_targets, perceptron_train_model)
    print("Combined network execution time : {0}".format(time.time() - tick))

    ph.plot_ae_perceptron_combined(perceptron_test_result, ae_test_result, combined_test_result)
