import numpy as np
from tensorflow import keras
from keras import layers
import DataSets.KerasFormatDS as ds_format
import DataSets.DatasetParameters as ds_parameters
import Tools.PlotHelper as ph

ds_param = ds_parameters.get_example_dataset()

def perceptron(hidden_layer_sizes, epochs =10):
    ds_train, ds_test = ds_format.classifier_dataset_column_inputs(ds_param.dataset_id, ds_param.feature_name, ds_param.batch_size)

    # Build model
    inputs = keras.Input(shape=ds_param.flat_input_size)

    x = layers.Rescaling(1.0 / 255)(inputs)
    for size in hidden_layer_sizes:
        x = layers.Dense(size, "relu", name="dense_1")(x)

    outputs = layers.Dense(ds_param.class_number, activation="softmax", name="dense_2")(x)

    model = keras.Model(inputs, outputs)

    # Config of model with losses and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Model training
    model.fit(
        ds_train,
        epochs=epochs,
        shuffle=True,
        validation_data=ds_test
    )

def perceptron_auto_encoder(hidden_layer_sizes, latent_space=10, epochs=10):
    ds_train, ds_test = ds_format.auto_encoder_dataset_column_inputs(ds_param.dataset_id, ds_param.feature_name, ds_param.batch_size)

    # Build model
    inputs = keras.Input(shape=ds_param.flat_input_size)
    x = layers.Rescaling(1.0 / 255)(inputs)

    # encoder block
    for index, size in enumerate(hidden_layer_sizes):
        x = layers.Dense(size, "relu", name="encoder_{0}".format(index))(x)

    # latent space
    x = layers.Dense(latent_space, 'relu', name='latent_space')(x)

    # decoder block
    for index, size in enumerate(reversed(hidden_layer_sizes)):
        x = layers.Dense(size, "relu", name="decoder_{0}".format(index))(x)

    outputs = layers.Dense(np.product(np.asarray(ds_param.input_shape)), activation="relu", name="outputs")(x)

    model = keras.Model(inputs, outputs)

    # Config of model with losses and metrics
    model.compile(
        optimizer=keras.optimizers.SGD(1e-3),
        loss="mean_squared_error"
    )

    # Model training
    model.fit(
        ds_train,
        epochs=epochs,
        shuffle=True,
        validation_data=ds_test
    )

    ph.plot_keras_auto_encoder_results(model, ds_test, 36)

def convolution(epochs=10):
    ds_train, ds_test = ds_format.convolution_dataset_image_inputs(ds_param.dataset_id, ds_param.feature_name, ds_param.batch_size)

    # Build model
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Convolution2D(3, 6, 2)(x) # (3, 12, 12)
    x = layers.MaxPool2D()(x) # (3, 6, 6)
    x = layers.Convolution2D(6, 2, 2)(x) # (6, 3, 3)
    x = layers.Flatten()(x)
    x = layers.Dense(6*3*3, "relu", name="middle_dense")(x)

    outputs = layers.Dense(ds_param.class_number, activation="softmax", name="outputs")(x)

    model = keras.Model(inputs, outputs)

    # Config of model with losses and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Model training
    model.fit(
        ds_train,
        epochs=epochs,
        shuffle=True,
        validation_data=ds_test
    )

def convolution_auto_encoder(input_shape, latent_space=10, epochs=10):
    ds_train, ds_test = ds_format.dataset_ae_convolution_dataset_image_inputs(ds_param.dataset_id, ds_param.feature_name, ds_param.batch_size)

    # Build model
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(encoder_inputs)

    x = layers.Convolution2D(3, 6, 2)(x)
    x = layers.Convolution2D(10, 2, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, "relu", name="encoder_dense")(x)

    x = layers.Dense(latent_space, "relu", name="latent_layer")(x)

    x = layers.Dense(250, "relu", name="decoder_dense")(x)
    x = layers.Reshape((5, 5, 10))(x)
    x = layers.Conv2DTranspose(10, 3, 2)(x)
    x = layers.Conv2DTranspose(3, 6, 2)(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3)(x)

    ae_model = keras.Model(encoder_inputs, decoder_outputs, name='ae')

    # Config of model with losses and metrics
    ae_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="mean_squared_error",
    )

    # Model training
    ae_model.fit(
        ds_train,
        epochs=epochs,
        shuffle=True,
        validation_data=ds_test
    )

    ph.plot_keras_auto_encoder_convolution_results(ae_model, ds_train, (28,28), 36)
