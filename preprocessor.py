import os
import time

import biosppy as biosppy
import numpy as np
import pandas as pd

# 300Hz sample rate
from logcreator.logcreator import Logcreator

SAMPLE_RATE = 300


def extract_mean_variance(sample):
    """
    Extracts the mean and variance heart beat of a given sample
    :return: mean_heart_beat, variance, heart_rate
    """

    # for now we just use default biosppy processing pipeline
    # TODO try to find best preprocessing: filtering, rpeak detection, ...

    # By default biosppy preprocessing returns 180 data points per heartbeat independent of the sample;
    # Thus this is for now the easiest method to get heartbeats with equal number of data points.
    ts, filtered, rpeaks, heartbeat_templates_ts, heartbeat_templates, heart_rate_ts, heart_rate = biosppy.signals.ecg.ecg(
        signal=sample, sampling_rate=SAMPLE_RATE, show=False)

    if heartbeat_templates.shape[0] == 180:
        raise ValueError('the length is not 180')

    mean_hb_graph = np.mean(heartbeat_templates, axis=0)
    var_hb_graph = np.var(heartbeat_templates, axis=0)
    mean_hb_rate = np.mean(heart_rate, axis=0)

    return mean_hb_graph, var_hb_graph, mean_hb_rate


def extract_features(x, x_name, extract_function, extracted_column_names):
    """
    General function to extract features and save them individually to a file.

    :param x: input data
    :param x_name: the name of the input data
    :param extract_function: function that extracts features for one sample of x
    :param extracted_column_names: the names of the extracted features
    :return:
    """
    start_time = time.time()
    Logcreator.info("\n", x_name, "start")

    # save index for later when saving the features
    index = x.index

    feature_list = []
    value_error_count = 0
    for i in range(0, x.shape[0]):
        if (i % int(x.shape[0] / 10)) == 0:
            elapsed_time = time.time() - start_time
            print(int(i / int(x.shape[0])*100), "% - samples processed:", i, "in %d [s]." % elapsed_time)

        sample = x.iloc[i]

        # shorten series to non nan values
        last_non_nan_idx = pd.Series.last_valid_index(sample)
        sample = sample[:last_non_nan_idx]

        # Reset the index because it causes unwanted effects in the library functions!
        sample = sample.reset_index(drop=True)

        # sample to array
        sample = sample.values

        try:
            extracted_values = extract_function(sample)
            feature_list.append(extracted_values)
        except ValueError as error:
            value_error_count = value_error_count + 1

    if value_error_count > 0:
        Logcreator.warn("Number of samples ignored:", value_error_count)

    data = pd.DataFrame(feature_list)
    data.columns = extracted_column_names

    for column_name in data:
        column = data[column_name]

        # unpack ndarrays if the given column contains ndarrays
        extracted = pd.DataFrame(column.values.tolist())

        # set column name if there is only one feature column
        if extracted.shape[1] == 1:
            extracted.columns = {column_name}

        extracted.set_index(index, inplace=True)
        extracted.to_csv(os.path.join("./data/extracted_features", x_name + "_" + column.name + ".csv"), index=True)

    total_elapsed_time = time.time() - start_time
    Logcreator.info("\nFeature extraction finished in %d [s]." % total_elapsed_time)


if __name__ == '__main__':
    # set n_rows to a integer for testing, to read only the top n-rows
    n_rows = None

    Logcreator.info("\nReading input files")

    start_time = time.time()

    x_train = pd.read_csv("./data/X_train.csv", index_col=0, nrows=n_rows, low_memory=False)
    y_train = pd.read_csv("./data/y_train.csv", index_col=0, nrows=n_rows, low_memory=False)
    x_test = pd.read_csv("./data/X_test.csv", index_col=0, nrows=n_rows, low_memory=False)

    # delete last column because in the last row it contains a string or something?!
    if n_rows is None:
        print("Wired string? in column x17978 row 5116:", x_train['x17978'][5116])
        del x_train['x17978']
        del x_test['x17978']

    # convert all columns to float32 - speeds up processing
    # INFO: we could also use float64 for more precision, but is it really needed?
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    elapsed_time = time.time() - start_time

    Logcreator.info("\nInput files read", "in %d [s]." % elapsed_time)

    # ----------------------------------------------------------------
    # extract features

    # extracting mean, variance and mean heart rate
    extracted_feature_names = ["mean", "variance", "mean-heart-rate"]
    # took 387s
    extract_features(x_train, "x_train", extract_mean_variance, extracted_feature_names)
    # took 256s
    extract_features(x_test, "x_test", extract_mean_variance, extracted_feature_names)
