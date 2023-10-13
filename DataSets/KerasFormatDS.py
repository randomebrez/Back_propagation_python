import keras.utils
import tensorflow as tf
import DataSets.OpenMLGateway as oml_gateway

def classifier_dataset_column_inputs(dataset_id, feature_name, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = oml_gateway.get_dataset(dataset_id, feature_name)

    dataset = tf.data.Dataset.from_tensor_slices((datas.astype("float32"), keras.utils.to_categorical(data_targets.values)))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test

def auto_encoder_dataset_column_inputs(dataset_id, feature_name, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = oml_gateway.get_dataset(dataset_id, feature_name)

    dataset = tf.data.Dataset.from_tensor_slices((datas.astype("float32"), datas.astype("float32")))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test

def convolution_dataset_image_inputs(dataset_id, feature_name, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = oml_gateway.get_dataset(dataset_id, feature_name)

    dataset = tf.data.Dataset.from_tensor_slices((datas.values.reshape((datas.shape[0], 28, 28)).astype("float32"), keras.utils.to_categorical(data_targets.values)))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test

def dataset_ae_convolution_dataset_image_inputs(dataset_id, feature_name, batch_size=1000, training_cutoff_percent = 0.8):
    datas, data_targets = oml_gateway.get_dataset(dataset_id, feature_name)

    reshaped_dataset = datas.values.reshape((datas.shape[0], 28, 28))
    dataset = tf.data.Dataset.from_tensor_slices((reshaped_dataset.astype("float32"), reshaped_dataset.astype("float32")))

    ds_train, ds_test = keras.utils.split_dataset(dataset, training_cutoff_percent)
    ds_train = ds_train.shuffle(buffer_size=1024).batch(batch_size)
    ds_test = ds_test.shuffle(buffer_size=1024).batch(batch_size)
    return ds_train, ds_test