import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_perceptron_result(pre_train_test_result, mean_cost_by_batch_epochs, mean_cost_epochs, post_train_test_result):
    x_test = np.arange(1, len(pre_train_test_result['accuracy']) + 1)
    x_train = np.arange(1, len(mean_cost_by_batch_epochs) + 1)
    x_train_mean = np.linspace(0, len(x_train), num=len(mean_cost_epochs))

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x_test, pre_train_test_result['accuracy'], 'r-')
    axs[0, 0].plot(x_test, post_train_test_result['accuracy'], 'b-')
    axs[0, 0].set_ylim(bottom=0)
    axs[0, 0].set_title('Accuracy on test batches before & after training')

    axs[1, 0].plot(x_test, pre_train_test_result['values'], 'r-')
    axs[1, 0].plot(x_test, post_train_test_result['values'], 'b-')
    axs[1, 0].plot(x_test, post_train_test_result['good_answers'], 'g-')
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].set_title('Output mean values on test batches before & after training')

    axs[0, 1].plot(x_test, pre_train_test_result['cost_function'], 'r-')
    axs[0, 1].plot(x_test, post_train_test_result['cost_function'], 'b-')
    axs[0, 1].set_ylim(bottom=0)
    axs[0, 1].set_title('Cost on test batches before & after training')

    axs[1, 1].plot(x_train, mean_cost_by_batch_epochs, 'b-')
    axs[1, 1].plot(x_train_mean, mean_cost_epochs, 'r-')
    axs[1, 1].set_title('Cost evolution during training')
    axs[1, 1].set_ylim(bottom=0)

    plt.show()

def plot_auto_encoder_results(network, training_inputs, pre_train_test_result, train_batch_costs, mean_batch_costs, post_train_test_result, number_to_compare=10):
    x_test = np.arange(1, len(pre_train_test_result['cost_function']) + 1)
    x_train = np.arange(1, len(train_batch_costs) + 1)
    x_train_mean = np.linspace(0, len(x_train), num=len(mean_batch_costs))

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_test, pre_train_test_result['cost_function'], 'r-')
    axs[0].plot(x_test, post_train_test_result['cost_function'], 'b-')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Cost on test batches before & after training')

    axs[1].plot(x_train, train_batch_costs, 'b-')
    axs[1].plot(x_train_mean, mean_batch_costs, 'r-')
    axs[1].set_title('Cost evolution during training')
    axs[1].set_ylim(bottom=0)

    # Compare images
    random_inputs = []

    # Select random image in a random batch
    for i in range(number_to_compare):
        random_batch = np.random.randint(0, training_inputs.shape[0])
        random_input_index = np.random.randint(0, training_inputs.shape[1])
        # Get images to compare
        random_inputs.append(training_inputs[random_batch, random_input_index])

    array_inp = np.array(random_inputs)
    # Compute result
    network.feed_forward(array_inp, False)
    network_outputs = network.get_outputs()

    # Create image bloc (original|sep|output)
    image_size = 28
    sep_size = 2
    plot_output = np.zeros((number_to_compare, image_size, 2 * image_size + sep_size))
    for i in range(number_to_compare):
        plot_output[i, :, 0:image_size] = array_inp[i].reshape((image_size, image_size))
        plot_output[i, :, image_size + sep_size:2 * image_size + sep_size] = network_outputs[i].reshape((image_size, image_size))

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
            image = plot_output[j + column_number * i]
            axs[i][j].imshow(image, cmap='gray')
    # Fill in last row
    for j in range(-delta):
        image = plot_output[j + column_number * (line_number - 1)]
        axs[line_number - 1][j].imshow(image, cmap='gray')

    plt.show()

def plot_ae_perceptron_combined(perceptron_test_result, ae_test_result, combined_test_result):
    x_test = np.arange(1, len(ae_test_result['cost_function']) + 1)

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(x_test, combined_test_result['accuracy'], 'b-')
    axs[0].plot(x_test, perceptron_test_result['accuracy'], 'r-')
    axs[0].set_ylim(bottom=0)
    axs[0].set_title('Accuracy on test batches for combined and perceptron only')

    axs[1].plot(x_test, combined_test_result['values'], 'b-')
    axs[1].plot(x_test, combined_test_result['good_answers'], color='b', marker='x')
    axs[1].plot(x_test, perceptron_test_result['values'], 'r-')
    axs[1].plot(x_test, perceptron_test_result['good_answers'], color='r', marker='x')
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Network output mean values')

    plt.show()

def plot_keras_auto_encoder_results(model, training_inputs, number_to_compare=9):
    # Select random image in a random batch
    random_inputs = np.zeros((number_to_compare, 784))

    numpy_ds = list(training_inputs.as_numpy_iterator())
    ds_size = np.shape(numpy_ds)
    # Select random image in a random batch
    for i in range(number_to_compare):
        batch_index = np.random.randint(0, ds_size[0])
        input_index = np.random.randint(0, ds_size[2])
        random_inputs[i] = numpy_ds[batch_index][1][input_index]

    sample = model(tf.constant(random_inputs))
    network_outputs = sample.numpy()
    # Create image bloc (original|sep|output)
    image_size = 28
    sep_size = 2
    plot_output = np.zeros((number_to_compare, image_size, 2 * image_size + sep_size))
    for i in range(number_to_compare):
        plot_output[i, :, 0:image_size] = random_inputs[i, :].reshape((image_size, image_size))
        plot_output[i, :, image_size + sep_size:2 * image_size + sep_size] = network_outputs[i].reshape((image_size, image_size))

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
            image = plot_output[j + column_number * i, :, :]
            axs[i][j].imshow(image, cmap='gray')
    # Fill in last row
    for j in range(-delta):
        image = plot_output[j + column_number * (line_number - 1), :, :]
        axs[line_number - 1][j].imshow(image, cmap='gray')

    plt.show()
