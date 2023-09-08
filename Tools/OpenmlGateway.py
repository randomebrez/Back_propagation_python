import openml
import pandas as pd
import numpy as np


def get_data_set_normalized(dataset_id, feature_class, normalization_constant, batch_size=1000, training_cutoff_percent=0.8):
    dataset = openml.datasets.get_dataset(dataset_id)
    datas, data_targets, z, column_indexes, = dataset.get_data(feature_class)

    input_batches, target_batches = batch_dataset_with_normalized_values(batch_size, datas, data_targets, normalization_constant)
    return split_train_test_batches(input_batches, target_batches, training_cutoff_percent)


def batch_dataset(batch_size, inputs, targets):
    input_batches = []
    target_batches = []
    ds_size = len(targets)
    current_index = 0
    while current_index < ds_size:
        max_index = min(current_index + batch_size, ds_size)

        input_batches.append(np.transpose(inputs.iloc[current_index:max_index].to_numpy()))

        target_batch = rework_targets(targets.iloc[current_index:max_index], 10)
        target_batches.append(np.transpose(target_batch.to_numpy()))

        current_index += batch_size

    return input_batches, target_batches


def batch_dataset_with_normalized_values(batch_size, inputs, targets, normalization_constant):
    input_batches = []
    target_batches = []
    ds_size = len(targets)
    current_index = 0
    while current_index < ds_size:
        max_index = min(current_index + batch_size, ds_size)

        input_batch = inputs.iloc[current_index:max_index] / normalization_constant
        input_batches.append(np.transpose(input_batch.to_numpy()))

        target_batch = rework_targets(targets.iloc[current_index:max_index], 10)
        target_batches.append(np.transpose(target_batch.to_numpy()))

        current_index += batch_size

    return input_batches, target_batches


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
    result = []
    for target in targets:
        target_value = np.zeros(output_size)
        target_value[int(target)] = 1
        result.append(pd.Series(target_value))
    return pd.DataFrame(result, index=targets.index)


def openml_test():
    datalist = openml.datasets.list_datasets(output_format="dataframe")
    datasets1 = datalist[datalist.NumberOfInstances > 10000]
    datasets2 = datalist.query('NumberOfInstances > 10000')  # same as above
    ds_row = datasets1.iloc[1]
    ds_id = int(ds_row["did"])
    ds = openml.datasets.get_dataset(ds_id)
    x, y, z, t = ds.get_data()  # x : datas with last column is the target | y : None | z : | t : x column indexes
    datas, targets, z1, column_indedexes = ds.get_data('class')  # x : without last column | targets : x last column column
    numpy_datas = datas.head(5).to_numpy()
    # print(datalist.columns)
    # print("")
    # print(datasets1.head(3))
    # print("")
    # print(ds_row)
    # print("")
    print(
        f"This is dataset '{ds.name}', the target feature is "
        f"'{ds.default_target_attribute}'"
    )
    print("")
    print(f"datas : {datas.head(3)}")
    print("")
    print(f"targets : {targets}")
    print("")
    print(f"column_indedexes : {column_indedexes}")
    print("")
    print(f"numpy : {numpy_datas}")
    print("")

