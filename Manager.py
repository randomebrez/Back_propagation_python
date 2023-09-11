import time
import numpy as np
import matplotlib.pyplot as plt
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


def plot_result(pre_train_test_result, train_batch_costs, mean_batch_costs, post_train_test_result):
    start = time.time()
    # Plotting
    # for i in range(10):
    #     random_batch = np.random.randint(0, len(training_inputs))
    #     random_line = np.random.randint(0, len(training_inputs[random_batch]))
    #     random_input = training_inputs[random_batch].iloc[random_line].to_numpy()
    #     network_output = network.feed_forward(random_input, False)
    #     compare_image(random_input, network_output)

    x_test = np.arange(1, len(pre_train_test_result['accuracy']) + 1)
    x_train = np.arange(1, len(train_batch_costs) + 1)
    x_train_mean = np.linspace(0, len(x_train), num=len(mean_batch_costs))

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x_test, pre_train_test_result['accuracy'], 'r-')
    axs[0].plot(x_test, post_train_test_result['accuracy'], 'b-')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Accuracy on test batches before & after training')
    axs[1].plot(x_test, pre_train_test_result['cost_function'], 'r-')
    axs[1].plot(x_test, post_train_test_result['cost_function'], 'b-')
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Cost on test batches before & after training')
    axs[2].plot(x_train, train_batch_costs, 'b')
    axs[2].plot(x_train_mean, mean_batch_costs, '-r')
    axs[2].set_title('Cost evolution during training')
    axs[2].set_ylim(bottom=0)

    tick = time.time()
    print('Plotting time : {0}'.format(tick - start))
    plt.show()


def compare_image(input, target):
    input.shape = (28, 28)
    target.shape = (28,28)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(input, cmap='gray')
    axs[1].imshow(target, cmap='gray')