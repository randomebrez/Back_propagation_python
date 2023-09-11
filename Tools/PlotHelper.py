import matplotlib.pyplot as plt
import numpy as np


def plot_perceptron_result(pre_train_test_result, train_batch_costs, mean_batch_costs, post_train_test_result):
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


def plot_auto_encoder_results(network, training_inputs, pre_train_test_result, train_batch_costs, mean_batch_costs, post_train_test_result, number_to_compare=10):
    x_test = np.arange(1, len(pre_train_test_result['cost_function']) + 1)
    x_train = np.arange(1, len(train_batch_costs) + 1)
    x_train_mean = np.linspace(0, len(x_train), num=len(mean_batch_costs))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_test, pre_train_test_result['cost_function'], 'r-')
    axs[0].plot(x_test, post_train_test_result['cost_function'], 'b-')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Cost on test batches before & after training')
    axs[1].plot(x_train, train_batch_costs, 'b')
    axs[1].plot(x_train_mean, mean_batch_costs, '-r')
    axs[1].set_title('Cost evolution during training')
    axs[1].set_ylim(bottom=0)

    # Compare images
    input_size = network.input_size
    random_inputs = np.zeros((input_size, number_to_compare))

    # Select random image in a random batch
    for i in range(number_to_compare):
        random_batch = np.random.randint(0, len(training_inputs))
        random_input_index = np.random.randint(0, np.shape(training_inputs[random_batch])[1])
        # Get images to compare
        random_inputs[:, i] = training_inputs[random_batch][:, random_input_index]

    # Compute result
    network.feed_forward(random_inputs, False)
    network_outputs = network.get_outputs()

    # Create image bloc (original|sep|output)
    random_inputs.shape = (28, 28, number_to_compare)
    network_outputs.shape = (28, 28, number_to_compare)
    image_separation = np.ones((28, 2))

    plot_output = np.zeros((28, 58, number_to_compare))
    for i in range(number_to_compare):
        plot_output[:, :, i] = np.hstack((random_inputs[:, :, i], np.hstack((image_separation, network_outputs[:, :, i]))))

    # Compute subplot size
    column_number = int(np.sqrt(number_to_compare))
    line_number = int(np.sqrt(number_to_compare))
    delta = column_number * column_number - number_to_compare
    if delta < 0:
        line_number += 1

    fig, axs = plt.subplots(line_number, column_number)
    fig.suptitle('Original images - Computed images')

    # Plot bloc images
    for i in range(column_number):
        for j in range(column_number):
            initial_image = plot_output[:, :, j + column_number * i]
            axs[i][j].imshow(initial_image, cmap='gray')
    # Fill in last row
    for j in range(-delta):
        initial_image = plot_output[:, :, j + column_number * (line_number - 1)]
        axs[line_number - 1][j].imshow(initial_image, cmap='gray')

    plt.show()
