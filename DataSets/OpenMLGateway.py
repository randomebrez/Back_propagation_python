import openml
import pandas as pd
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
