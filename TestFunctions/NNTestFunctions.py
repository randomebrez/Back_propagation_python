import numpy as np
import time
import Manager as manager
import DataSets.NNFormatDS as ds_format
import TestFunctions.NNModelBuilder as model_builder
import DataSets.DatasetParameters as ds_parameters
import Tools.Computations as computer
import Class.TrainingModel as model
import Tools.PlotHelper as ph

ds_param = ds_parameters.get_example_dataset()

def perceptron(hidden_layer_sizes, epochs=10):
    # Fetch dataset
    ds_inputs, ds_targets = ds_format.get_column_data_set_normalized(ds_param.dataset_id, ds_param.feature_name, ds_param.class_number, ds_param.normalization_constant, batch_size=ds_param.batch_size)
    # Build network
    network = model_builder.perceptron(ds_param.flat_input_size, ds_param.class_number, hidden_layer_sizes)
    # Build training model
    model_parameters = model.ModelParameters(
        computer.cross_entropy,
        computer.cross_entropy_derivative,
        initial_learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_steps=5,
        epochs=epochs)

    start = time.time()
    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_targets, network, model_parameters)
    train_results = manager.train_network(ds_inputs, ds_targets, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_targets, network, model_parameters)

    # Plot results
    ph.plot_perceptron_result(pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test)

    tick = time.time()
    print("Perceptron network execution time : {0}".format(tick - start))
    return network

def auto_encoder(hidden_layer_sizes, latent_space=10, epochs=10):
    # Fetch dataset
    ds_inputs, ds_targets = ds_format.get_column_data_set_normalized(ds_param.dataset_id, ds_param.feature_name, ds_param.class_number, ds_param.normalization_constant, batch_size=ds_param.batch_size)
    # Build network
    network = model_builder.auto_encoder(ds_param.flat_input_size, ds_param.class_number, hidden_layer_sizes, latent_space)
    # Build training model
    model_parameters = model.ModelParameters(
        computer.mean_square_error,
        computer.distance_get,
        initial_learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_steps=5,
        epochs=epochs)

    start = time.time()
    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_inputs, network, model_parameters, with_details=False)
    train_results = manager.train_network(ds_inputs, ds_inputs, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_inputs, network, model_parameters, with_details=False)

    # Plot results
    ph.plot_auto_encoder_results(network, ds_inputs[0], pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test, 36)

    tick = time.time()
    print("Auto_encoder network execution time : {0}".format(tick - start))
    return network

def convolution(epochs=10):
    # Fetch dataset
    ds_inputs, ds_targets = ds_format.get_image_data_set_normalized(ds_param.dataset_id, ds_param.feature_name, ds_param.class_number, ds_param.normalization_constant, batch_size=ds_param.batch_size)
    # Build network
    network = model_builder.convolution(ds_param.input_shape, ds_param.class_number)
    # Build training model
    model_parameters = model.ModelParameters(
        computer.cross_entropy,
        computer.cross_entropy_derivative,
        initial_learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_steps=10,
        epochs=epochs)

    start = time.time()
    # Run model
    pre_train_result = manager.test_network(ds_inputs, ds_targets, network, model_parameters)
    train_results = manager.train_network(ds_inputs, ds_targets, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_targets, network, model_parameters)

    # Plot results
    ph.plot_perceptron_result(pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test)

    tick = time.time()
    print("Convolution network execution time : {0}".format(tick - start))
    return network

def perceptron_ae_combined(perceptron_hidden_layer_sizes, ae_hidden_layer_sizes, ae_latent_space=10, epochs=10):
    ds_inputs, ds_targets = ds_format.get_column_data_set_normalized(ds_param.dataset_id, ds_param.feature_name, ds_param.class_number, ds_param.normalization_constant, batch_size=ds_param.batch_size)

    # Perceptron
    perceptron_network = model_builder.perceptron(ds_param.flat_input_size, ds_param.class_number, perceptron_hidden_layer_sizes)
    perceptron_model_parameters = model.ModelParameters(
        computer.cross_entropy,
        computer.cross_entropy_derivative,
        initial_learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_steps=5,
        epochs=epochs)

    start = time.time()
    # training
    manager.train_network(ds_inputs, ds_targets, perceptron_network, perceptron_model_parameters)
    perceptron_test_result = manager.test_network(ds_inputs, ds_targets, perceptron_network, perceptron_model_parameters)
    tick = time.time()
    print("Perceptron network execution time : {0}".format(tick - start))

    # Auto_encoding
    ae_network = model_builder.auto_encoder(ds_param.flat_input_size, ds_param.class_number, ae_hidden_layer_sizes, ae_latent_space)
    ae_model_parameters = model.ModelParameters(
        computer.mean_square_error,
        computer.distance_get,
        initial_learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_steps=10,
        epochs=epochs)

    tick = time.time()
    # training
    manager.train_network(ds_inputs, ds_inputs, ae_network, ae_model_parameters)
    ae_test_result = manager.test_network(ds_inputs, ds_inputs, ae_network, ae_model_parameters)
    print("Auto_encoder network execution time : {0}".format(time.time() - tick))

    # Combined
    ae_perceptron = model_builder.perceptron(ds_param.flat_input_size, ds_param.class_number, perceptron_hidden_layer_sizes)

    tick = time.time()
    train_inputs, test_inputs = ds_inputs[0], ds_inputs[1]
    ae_train_outputs = np.zeros(train_inputs.shape)
    ae_test_outputs = np.zeros(test_inputs.shape)

    # Compute using auto-encoding
    for i in range(train_inputs.shape[0]):
        ae_network.feed_forward(train_inputs[i], False)
        ae_train_outputs[i] = ae_network.get_outputs()
    for i in range(test_inputs.shape[0]):
        ae_network.feed_forward(test_inputs[i], False)
        ae_test_outputs[i] = ae_network.get_outputs()
    ae_output_ds = [ae_train_outputs, ae_test_outputs]

    # training
    manager.train_network(ae_output_ds, ds_targets, ae_perceptron, perceptron_model_parameters)
    ae_perceptron_test_result = manager.test_network(ae_output_ds, ds_targets, ae_perceptron, perceptron_model_parameters)
    print("Combined network execution time : {0}".format(time.time() - tick))

    ph.plot_ae_perceptron_combined(perceptron_test_result, ae_test_result, ae_perceptron_test_result)

def auto_encoder_convolution(latent_space, epochs=10):
    ds_inputs, ds_targets = ds_format.get_image_data_set_normalized(ds_param.dataset_id, ds_param.feature_name, ds_param.class_number, ds_param.normalization_constant, batch_size=ds_param.batch_size)
    # Build network
    network = model_builder.ae_convolution(ds_param.input_shape, latent_space)
    # Build training model
    model_parameters = model.ModelParameters(
        computer.mean_square_error,
        computer.distance_get,
        initial_learning_rate=0.01,
        final_learning_rate=0.001,
        learning_rate_steps=10,
        epochs=epochs)

    start = time.time()
    # Run model
    #pre_train_result = manager.test_network(ds_inputs, ds_inputs, network, model_parameters, with_details=False)
    train_results = manager.train_network(ds_inputs, ds_inputs, network, model_parameters)
    post_train_test = manager.test_network(ds_inputs, ds_inputs, network, model_parameters, with_details=False)

    # Plot results
    #ph.plot_perceptron_result(pre_train_result, train_results['batch_costs'], train_results['mean_batch_costs'], post_train_test)

    tick = time.time()
    print("Auto-encoder convolution network execution time : {0}".format(tick - start))
    return network
