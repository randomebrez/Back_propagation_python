import time
import numpy as np
import Tools.OpenmlGateway as openml_gateway


def get_column_dataset(dataset_id, feature_class, class_number, normalization_constant, batch_size=100):
    # Call to openML server
    ds_inputs, ds_targets = openml_gateway.get_column_data_set_normalized(dataset_id, feature_class, class_number, normalization_constant, batch_size=batch_size)

    input_shape = ds_inputs[1].shape[2:]
    output_shape = ds_targets[1].shape[2:]
    return ds_inputs, ds_targets, input_shape, output_shape


def get_image_dataset(dataset_id, feature_class, class_number, normalization_constant, batch_size=100):
    # Call to openML server
    ds_inputs, ds_targets = openml_gateway.get_image_data_set_normalized(dataset_id, feature_class, class_number, normalization_constant, batch_size=batch_size)

    input_shape = ds_inputs[1].shape[2:]
    output_shape = ds_targets[1].shape[2:]
    return ds_inputs, ds_targets, input_shape, output_shape


def train_network(ds_inputs, ds_targets, network, train_model):
    tick = time.time()

    # Split dataset
    training_inputs, training_targets = ds_inputs[0], ds_targets[0]

    # Train network
    train_batch_cost_fct = []
    train_batch_mean_cost_fct = []

    for i in range(train_model.epochs):
        cost_function = network.train(training_inputs, training_targets, train_model)
        train_batch_cost_fct = np.concatenate((train_batch_cost_fct, cost_function))
        mean_cost_fct = np.mean(cost_function)
        train_batch_mean_cost_fct.append(mean_cost_fct)
        print("Run {0} done. Mean error : {1} -  Execution time : {2}".format(i + 1, round(float(mean_cost_fct), 5), round(time.time() - tick, 2)))
        training_inputs, training_targets = shuffle_training_datas(training_inputs, training_targets)
        tick = time.time()

    print('Execution time of \'train_network\' : {0}'.format(round(time.time() - tick, 2)))

    return {
        'batch_costs': train_batch_cost_fct,
        'mean_batch_costs': train_batch_mean_cost_fct
    }

def shuffle_training_datas(training_inputs, training_targets):
    permutations = np.random.permutation(training_inputs.shape[0])
    return training_inputs[permutations], training_targets[permutations]

def test_network(ds_inputs, ds_targets, network, train_model, with_details=True):
    tick = time.time()

    # Split dataset
    test_inputs, test_targets = ds_inputs[1], ds_targets[1]

    # Run model
    test_result = network.test(test_inputs, test_targets, train_model, with_details)

    print('Execution time of \'test_network\' : {0}'.format(round(time.time() - tick, 2)))

    return test_result
