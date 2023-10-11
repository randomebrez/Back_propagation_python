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
    mean_cost_by_batch_epochs = []
    mean_cost_epochs = []

    for i in range(train_model.epochs):
        cost_all_batch = network.train(training_inputs, training_targets, train_model)
        mean_cost_by_batch_epochs = np.concatenate((mean_cost_by_batch_epochs, cost_all_batch))
        mean_cost_epochs.append(np.mean(cost_all_batch))
        print("Run {0} done. Mean error : {1} -  Execution time : {2}".format(i + 1, round(float(mean_cost_epochs[-1]), 5), round(time.time() - tick, 2)))
        training_inputs, training_targets = shuffle_training_datas(training_inputs, training_targets)
        tick = time.time()

    print('Execution time of \'train_network\' : {0}'.format(round(time.time() - tick, 2)))

    return {
        'batch_costs': mean_cost_by_batch_epochs,
        'mean_batch_costs': mean_cost_epochs
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
