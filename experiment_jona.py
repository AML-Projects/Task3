import biosppy as biosppy
import pandas as pd

if __name__ == '__main__':
    # only read top n-rows for faster manual analysis
    n_rows = 10
    x_train = pd.read_csv("./data/X_train.csv", index_col=0, nrows=n_rows)
    y_train = pd.read_csv("./data/y_train.csv", index_col=0, nrows=n_rows)
    x_test = pd.read_csv("./data/X_test.csv", index_col=0, nrows=n_rows)

    # 300Hz sample rate
    SAMPLE_RATE = 300
    for i in range(0, x_train.shape[0]):
        sample = x_train.iloc[i]
        last_non_nan_idx = pd.Series.last_valid_index(sample)
        sample = sample[:last_non_nan_idx]
        biosppy.signals.ecg.ecg(signal=sample, sampling_rate=SAMPLE_RATE, show=True)
