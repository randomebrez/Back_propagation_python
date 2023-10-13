import numpy as np
import pandas as pd
import DataSets.OpenMLGateway as oml_gateway

# datas_shape = (instance, input_values) | data_targets_shape = (instance, )
def get_column_data_set_normalized(dataset_id, feature_name, class_number, normalization_constant, batch_size=1000, training_cutoff_percent=0.8):
    datas, data_targets = oml_gateway.get_dataset(dataset_id, feature_name)
    ds_size = data_targets.shape[0]
    batch_number = datas.shape[0] // batch_size

    input_batches = np.empty((batch_number, batch_size, datas.shape[1]))
    target_batches = np.empty((batch_number, batch_size, class_number))

    batch_index = 0
    while batch_index < batch_number:
        min_index = batch_size * batch_index
        max_index = min(min_index + batch_size, ds_size)

        input_batch = datas.iloc[min_index:max_index] / normalization_constant
        input_batches[batch_index] = input_batch.to_numpy()

        target_batch = rework_targets(data_targets.iloc[min_index:max_index], class_number)
        target_batches[batch_index] = target_batch

        batch_index += 1

    return split_train_test_batches(input_batches, target_batches, training_cutoff_percent)

def get_image_data_set_normalized(dataset_id, feature_name, class_number, normalization_constant, batch_size=1000, training_cutoff_percent=0.8):
    datas, data_targets = oml_gateway.get_dataset(dataset_id, feature_name)
    ds_size = data_targets.shape[0]

    batch_number = datas.shape[0] // batch_size
    # Todo : see on rectangle image how to extract info
    image_size = int(np.sqrt(datas.shape[1]))
    # Todo : see on a dataset of image with multiple channels how to extract info
    depth = 1

    input_batches = np.empty((batch_number, batch_size, depth, image_size, image_size))
    target_batches = np.empty((batch_number, batch_size, class_number))

    batch_index = 0
    while batch_index < batch_number:
        min_index = batch_size * batch_index
        max_index = min(min_index + batch_size, ds_size)

        input_batch = datas.iloc[min_index:max_index] / normalization_constant
        input_batches[batch_index] = input_batch.to_numpy().reshape((batch_size, depth, image_size, image_size))

        target_vectors = rework_targets(data_targets.iloc[min_index:max_index], class_number)
        target_batches[batch_index] = target_vectors

        batch_index += 1

    return split_train_test_batches(input_batches, target_batches, training_cutoff_percent)

# Tools
def split_train_test_batches(input_batches, target_batches, training_cutoff_percent):
    training_cutoff = int(training_cutoff_percent * len(input_batches))

    # Training
    training_inputs = input_batches[:training_cutoff]
    training_targets = target_batches[:training_cutoff]

    # Test
    test_inputs = input_batches[training_cutoff:]
    test_targets = target_batches[training_cutoff:]

    return [training_inputs, test_inputs], [training_targets, test_targets]

def rework_targets(targets, output_size):
    result = np.zeros((targets.shape[0], output_size))
    if type(targets) is pd.DataFrame:
        for index, target in enumerate(targets.iloc[1:, 0]):
            result[index, int(target)] = 1
    else:
        for index, target in enumerate(targets):
            result[index, int(target)] = 1

    return result