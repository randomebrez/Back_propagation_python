import TestFunctions.KerasTestFunctions as keras_test
import TestFunctions.NNTestFunctions as nn_test


launch_NN = 1

perceptron_hidden_layer_sizes = [700]
ae_hidden_layers = [700]
ae_latent_space = 50

if (launch_NN):
    print("NN launched")
    nn_test.perceptron(perceptron_hidden_layer_sizes)
    # nn_test.auto_encoder(ae_hidden_layers, ae_latent_space)
    # nn_test.perceptron_ae_combined(perceptron_hidden_layer_sizes, ae_hidden_layers, ae_latent_space)
    # nn_test.convolution()
else:
    print("Keras launched")
    # keras_test.perceptron(perceptron_hidden_layer_sizes)
    #keras_test.perceptron_auto_encoder(ae_hidden_layers, ae_latent_space)
    keras_test.convolution()
    #keras_test.convolution_auto_encoder((28, 28, 1), 50)

