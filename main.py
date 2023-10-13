import TestLibraryFunction as tests
import Tools.KerasGateway as k_gateway


launch_my_tool = 0

if (launch_my_tool):
    print("Tool launched")
    # tests.test_perceptron_ae_combined()
    # tests.test_perceptron()
    # tests.test_auto_encoder()
    tests.test_conv_net()
else:
    print("Keras launched")
    # k_gateway.perceptron()
    # k_gateway.auto_encoder()
    k_gateway.convolution()
    # k_gateway.auto_encoder_convolution((28, 28, 1))

