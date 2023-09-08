import time
import numpy as np
import matplotlib.pyplot as plt
import Tools.OpenmlGateway as openml_gateway
from Class import NetworkBuilder as builder


def get_dataset(dataset_id, feature_class, normalization_constant, batch_size=100):
    return openml_gateway.get_data_set_normalized(dataset_id, feature_class, normalization_constant, batch_size=batch_size)


def back_prop_batch_on_dataset(ds_inputs, ds_targets, hidden_layer_sizes, loss_function, loss_function_derivative,
                               initial_learning_rate=0.1, final_learning_rate=0.01, learning_rate_update_number=10, epochs=100):
    start = time.time()

    # Get inputs
    training_inputs, training_targets = ds_inputs[0], ds_targets[0]
    test_inputs, test_targets = ds_inputs[1], ds_targets[1]

    # Build network
    input_size = np.shape(training_inputs[0])[0]
    output_size = np.shape(training_targets[0])[0]
    network_builder = builder.NetworkBuilder(input_size, output_size)
    #network = network_builder.build_dense_network(hidden_layer_sizes)
    network = network_builder.build_dense_network_softmax_output(hidden_layer_sizes)

    # Run network
    pre_train_test_result = network.test(test_inputs, test_targets, loss_function, loss_function_derivative)

    train_batch_cost_fct = []
    for i in range(epochs):
        cost_function = network.train(training_inputs, training_targets, loss_function, loss_function_derivative, initial_learning_rate, final_learning_rate, learning_rate_update_number)
        train_batch_cost_fct = np.concatenate((train_batch_cost_fct, cost_function))
        print("Run {0} done. Mean error : {1}".format(i + 1, np.mean(cost_function)))

    post_train_test_result = network.test(test_inputs, test_targets, loss_function, loss_function_derivative)

    # Plotting
    # for i in range(10):
    #     random_batch = np.random.randint(0, len(training_inputs))
    #     random_line = np.random.randint(0, len(training_inputs[random_batch]))
    #     random_input = training_inputs[random_batch].iloc[random_line].to_numpy()
    #     network_output = network.feed_forward(random_input, False)
    #     compare_image(random_input, network_output)

    x_test = np.arange(1, len(pre_train_test_result['accuracy']) + 1)
    x_train = np.arange(1, len(train_batch_cost_fct) + 1)

    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x_test, pre_train_test_result['accuracy'], 'r-')
    axs[0].plot(x_test, post_train_test_result['accuracy'], 'b-')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Accuracy on test batches before & after training')
    axs[1].plot(x_test, pre_train_test_result['cost_function'], 'r-')
    axs[1].plot(x_test, post_train_test_result['cost_function'], 'b-')
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Cost on test batches before & after training')
    axs[2].plot(x_train, train_batch_cost_fct, 'b')
    axs[2].set_title('Cost evolution during training')
    axs[2].set_ylim(bottom=0)

    tick = time.time()
    print('Execution time of \'back_prop_batch_on_dataset\' : {0}'.format(tick - start))
    plt.show()

def compare_image(input, target):
    input.shape = (28, 28)
    target.shape = (28,28)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(input, cmap='gray')
    axs[1].imshow(target, cmap='gray')