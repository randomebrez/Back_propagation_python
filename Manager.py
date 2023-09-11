import time
import numpy as np
import Tools.OpenmlGateway as openml_gateway


def get_dataset(dataset_id, feature_class, normalization_constant, batch_size=100):
    # Call to openML server
    ds_inputs, ds_targets = openml_gateway.get_data_set_normalized(dataset_id, feature_class, normalization_constant, batch_size=batch_size)
    return ds_inputs, ds_targets


def train_network(ds_inputs, ds_targets, network, train_model):
    start = time.time()

    # Split dataset
    training_inputs, training_targets = ds_inputs[0], ds_targets[0]
    test_inputs, test_targets = ds_inputs[1], ds_targets[1]

    # Test network before training
    pre_train_test_result = network.test(test_inputs, test_targets, train_model)

    # Train network
    train_batch_cost_fct = []
    train_batch_mean_cost_fct = []

    for i in range(train_model.epochs):
        cost_function = network.train(training_inputs, training_targets, train_model)
        train_batch_cost_fct = np.concatenate((train_batch_cost_fct, cost_function))
        mean_cost_fct = np.mean(cost_function)
        train_batch_mean_cost_fct.append(mean_cost_fct)
        print("Run {0} done. Mean error : {1}".format(i + 1, mean_cost_fct))

    # Test network after training
    post_train_test_result = network.test(test_inputs, test_targets, train_model)

    tick = time.time()
    print('Execution time of \'back_prop_batch_on_dataset\' : {0}'.format(tick - start))

    return {
        'pre_train_test': pre_train_test_result,
        'batch_costs': train_batch_cost_fct,
        'mean_batch_costs': train_batch_mean_cost_fct,
        'post_train_test': post_train_test_result
    }
