import numpy as np
from scipy.signal import fftconvolve


def input_compute_shape(input_batch, kernel_size_x):
    (batch_size, depth, x_size, y_size) = input_batch.shape
    index_shift = (x_size + kernel_size_x)
    shaped_input = np.zeros((depth, batch_size * index_shift - kernel_size_x, y_size))

    for i in range(batch_size):
        index_start = i * index_shift
        shaped_input[:, index_start:index_start + x_size] = input_batch[i]

    return shaped_input


def input_conv_res_shape(convolution_result, batch_size, kernel_size, stride, padding, input_shape, output_shape):
    padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
    kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
    stride_x, stride_y = stride[1], stride[2]
    input_depth, input_shape_x, input_shape_y = input_shape[0], input_shape[1], input_shape[2]

    # Truncate & Slice according to z with only full overlap (for each filter)
    full_overlap_conv_res = convolution_result[input_depth - 1::input_depth]

    # Split above result into batch_size array of single outputs
    conv_res_split = np.array_split(full_overlap_conv_res, batch_size, axis=1)

    # Indices for truncating single images according to padding
    x_to_remove_each_side = (kernel_size_x - 1) - padding_x
    y_to_remove_each_side = (kernel_size_y - 1) - padding_y

    results = np.empty((batch_size, output_shape[0], output_shape[1], output_shape[2]))
    for i in range(batch_size):
        single_image = conv_res_split[i]
        # Remove 'marker' column of the convolution result (except for last one)
        if i != batch_size - 1:
            single_image = single_image[:, :-1]

        # Truncate image according to padding
        single_image = single_image[:, x_to_remove_each_side:-x_to_remove_each_side,
                       y_to_remove_each_side::-y_to_remove_each_side]

        # Slice single image according to stride
        sliced_conv_res = single_image[:, ::stride_x, ::stride_y]

        # Save
        results[i] = sliced_conv_res
    return results


def dw_compute_shape(previous_layer_activation, back_prop_activated_values, kernel_size, stride, padding, filter_number, input_shape):
    input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
    padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
    kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
    stride_x, stride_y = stride[1], stride[2]
    batch_size, bp_x, bp_y = back_prop_activated_values.shape[0], back_prop_activated_values.shape[2], back_prop_activated_values.shape[3]

    bp_shape_depth = (filter_number - 1) * (input_depth - 1) + filter_number
    # size of 1 bloc : InterestingValues | (kernel_size - 1) 0
    bp_x_shift = bp_x + (stride_x - 1) * (bp_x - 1) + kernel_size_x - 1
    bp_shape_x = bp_x_shift * batch_size - (kernel_size_x - 1)
    bp_shape_y = (bp_y + (stride_y - 1) * (bp_y - 1))
    shaped_bp_values = np.zeros((bp_shape_depth, bp_shape_x, bp_shape_y))

    input_x_shift = input_x + 2 * padding_x
    input_shape_x = input_x_shift * batch_size - 2 * padding_x
    shaped_input = np.zeros((input_depth, input_shape_x, input_y))

    for i in range(batch_size):
        bp_index_start = i * bp_x_shift
        bp_index_max = bp_index_start + bp_x_shift - (kernel_size_x - 1)
        shaped_bp_values[::input_depth, bp_index_start:bp_index_max:stride_x, ::stride_y] = back_prop_activated_values[i]

        input_index_start = i * input_x_shift
        input_index_max = input_index_start + input_x_shift - 2 * padding_x
        shaped_input[:, input_index_start:input_index_max] = previous_layer_activation[i]

    return shaped_input, shaped_bp_values


def dw_conv_res_shape(convolution_result, back_prop_activated_values, kernel_size, stride, padding, filter_number,
                      input_shape):
    input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
    kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
    stride_x, stride_y = stride[1], stride[2]
    padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
    batch_size, bp_x, bp_y = back_prop_activated_values.shape[0], back_prop_activated_values.shape[2], \
                             back_prop_activated_values.shape[3]

    # Split depth wise
    depth_split = np.array_split(convolution_result, filter_number, axis=0)

    # Indices for truncating single images according to padding
    x_to_remove_each_side = batch_size * (bp_x + (stride_x - 1) * (bp_x - 1) + kernel_size_x - 1) - (
                kernel_size_x - 1) - padding_x - 1
    y_to_remove_each_side = (bp_y + (stride_y - 1) * (bp_y - 1) - 1) - padding_y

    filter_results = np.zeros((input_depth * filter_number, kernel_size_x, kernel_size_y))
    for i in range(filter_number):
        filter_result = depth_split[i]

        # Remove useless values row/col wise
        temp = filter_result[:, x_to_remove_each_side:-x_to_remove_each_side,
               y_to_remove_each_side:-y_to_remove_each_side]

        # Mean on each input of the batch
        filter_results[i * input_depth:(i + 1) * input_depth] = temp / batch_size

    return filter_results


def de_compute_shape(filters, back_prop_activated_values, filter_number, kernel_size, stride, input_shape):
    stride_x, stride_y = stride[1], stride[2]
    input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
    kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]
    batch_size, bp_x, bp_y = back_prop_activated_values.shape[0], back_prop_activated_values.shape[2], back_prop_activated_values.shape[3]

    # Mirror filter block
    filter_mirror = np.flip(np.flip(np.flip(filters, 0), 1), 2)

    # size of 1 bloc : InterestingValues | kernel_size 0
    x_shift = bp_x + (bp_x - 1) * (stride_x - 1) + kernel_size_x - 1

    bp_shape_depth = (filter_number - 1) * (input_depth - 1) + filter_number
    bp_shape_x = x_shift * batch_size - (kernel_size_x - 1)
    bp_shape_y = bp_y + (bp_y - 1) * (stride_y - 1)
    shaped_bp_values = np.zeros((bp_shape_depth, bp_shape_x, bp_shape_y))

    for i in range(batch_size):
        index_start = i * x_shift
        index_max = index_start + x_shift - (kernel_size_x - 1)
        shaped_bp_values[::input_depth, index_start:index_max:stride_x, ::stride_y] = np.flip(back_prop_activated_values[i], 0)

    return filter_mirror, shaped_bp_values


def de_conv_res_shape(convolution_result, batch_size, kernel_size, filter_number, padding, input_shape):
    input_depth, input_x, input_y = input_shape[0], input_shape[1], input_shape[2]
    padding_depth, padding_x, padding_y = padding[0], padding[1], padding[2]
    kernel_size_x, kernel_size_y = kernel_size[1], kernel_size[2]

    # Truncate depth wise (only overlap of feature map with full input depth)
    z_to_remove = (filter_number - 1) * input_depth
    depth_truncate = convolution_result[z_to_remove:-z_to_remove]

    single_split = np.array_split(depth_truncate, int(batch_size), axis=1)
    mean_values = np.sum(single_split, axis=0)

    # Indices for truncating single images according to padding
    x_to_remove_each_side = kernel_size_x - padding_x - 1
    y_to_remove_each_side = kernel_size_y - padding_y - 1
    truncated_mean_values = mean_values[:, x_to_remove_each_side:-x_to_remove_each_side, y_to_remove_each_side:-y_to_remove_each_side]

    return truncated_mean_values / (batch_size * filter_number)


# Convolve using FFT
def convolve_filters(kernel, shaped_input_batch):
    return fftconvolve(kernel, shaped_input_batch)
