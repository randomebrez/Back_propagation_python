import numpy as np

def get_example_dataset():
    return DS_Parameters(40996, (1, 28, 28), 'class', 10, 255, 100)

class DS_Parameters:
    def __init__(self, dataset_id, input_shape, feature_name, class_number, normalization_constant, batch_size):
        self.dataset_id = dataset_id
        self.input_shape = input_shape
        self.flat_input_size = np.product(np.asarray(input_shape))
        self.feature_name = feature_name
        self.class_number = class_number
        self.normalization_constant = normalization_constant
        self.batch_size = batch_size
