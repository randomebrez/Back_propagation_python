import keras.utils
import openml
import pandas as pd
import tensorflow as tf
import numpy as np
import os


working_directory = r"C:\Users\nico-\Desktop\OpenML_Datasets"

def get_dataset(dataset_id, feature_class):
    dataset_directory = r"{0}\{1}".format(working_directory, dataset_id)
    file_path_inputs, file_path_targets = r"{0}\inputs.txt".format(dataset_directory), r"{0}\targets.txt".format(dataset_directory)

    if (os.path.exists(dataset_directory)):
        datas = pd.read_csv(file_path_inputs)
        data_targets = pd.read_csv(file_path_targets)
    else:
        dataset = openml.datasets.get_dataset(dataset_id)
        # Get dataset from openML
        datas, data_targets, z, column_indexes, = dataset.get_data(feature_class)
        # Create directory
        os.mkdir(dataset_directory)
        # Write inputs
        new_input_file = open(file_path_inputs, mode='w')
        new_input_file.write(datas.to_csv(index=False))
        new_input_file.close()
        #Write targets
        new_target_file = open(file_path_targets, mode='w')
        new_target_file.write(data_targets.to_csv(index=False))
        new_target_file.close()
    return datas, data_targets


# Shape datas for my tool models (ie numpy arrays)
# datas_shape = (instance, input_values) | data_targets_shape = (instance, )
def get_column_data_set_normalized(dataset_id, feature_class, class_number, normalization_constant, batch_size=1000, training_cutoff_percent=0.8):
    datas, data_targets = get_dataset(dataset_id, feature_class)
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

def get_image_data_set_normalized(dataset_id, feature_class, class_number, normalization_constant, batch_size=1000, training_cutoff_percent=0.8):
    datas, data_targets = get_dataset(dataset_id, feature_class)
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

# Shape datas for a Keras model (ie as Tensors)
def get_keras_dataset(dataset_id, feature_class, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = get_dataset(dataset_id, feature_class)

    dataset = tf.data.Dataset.from_tensor_slices((datas.astype("float32"), keras.utils.to_categorical(data_targets.values)))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test

def get_keras_dataset_auto_encoder(dataset_id, feature_class, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = get_dataset(dataset_id, feature_class)

    dataset = tf.data.Dataset.from_tensor_slices((datas.astype("float32"), datas.astype("float32")))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test

def get_keras_dataset_convolution(dataset_id, feature_class, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = get_dataset(dataset_id, feature_class)

    dataset = tf.data.Dataset.from_tensor_slices((datas.values.reshape((datas.shape[0], 28, 28)).astype("float32"), keras.utils.to_categorical(data_targets.values)))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test

def get_keras_dataset_ae_convolution(dataset_id, feature_class, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = get_dataset(dataset_id, feature_class)

    reshaped_dataset = datas.values.reshape((datas.shape[0], 28, 28))
    dataset = tf.data.Dataset.from_tensor_slices((reshaped_dataset.astype("float32"), reshaped_dataset.astype("float32")))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test