import warnings

import biosppy as biosppy
import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

# 300Hz sample rate
SAMPLE_RATE = 300


def analysis_with_biosppy_lib(sample):
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = biosppy.signals.ecg.ecg(
        signal=sample, sampling_rate=SAMPLE_RATE, show=True)

    # we can extract rpeaks with different methods
    rpeaks_es = biosppy.signals.ecg.engzee_segmenter(sample, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_hs = biosppy.signals.ecg.hamilton_segmenter(sample.values, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_cs = biosppy.ecg.christov_segmenter(sample, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_gs = biosppy.ecg.gamboa_segmenter(sample, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_ssfs = biosppy.ecg.ssf_segmenter(sample, sampling_rate=SAMPLE_RATE)[0]

    # get heartbeat templates
    heart_beat_templates = biosppy.signals.ecg.extract_heartbeats(signal=sample,
                                                                  rpeaks=rpeaks,
                                                                  sampling_rate=SAMPLE_RATE)

    # heartbeat templates -> Problem they have different length from one sample to another sample!
    print(templates.shape)
    if False:
        plt.plot(ts, filtered)
        plt.title("class " + str(label))
        plt.grid()
        plt.show()

    pass


def analysis_with_neurokit2(sample):
    # we have to reset the index, otherwise ecg_process returns unwanted results
    sample = sample.reset_index(drop=True)
    ecg_df, r_peaks = nk.ecg_process(ecg_signal=sample, sampling_rate=SAMPLE_RATE)

    # binary series describing where the peaks are
    ecg_df["ECG_P_Peaks"]
    ecg_df["ECG_Q_Peaks"]
    ecg_df["ECG_R_Peaks"]
    ecg_df["ECG_S_Peaks"]
    ecg_df["ECG_T_Peaks"]
    # plot
    nk.ecg_plot(ecg_signals=ecg_df, sampling_rate=SAMPLE_RATE, show_type='full')
    plt.show()

    # we can also just get the rpeaks
    r_peaks2 = nk.ecg_findpeaks(ecg_cleaned=ecg_df["ECG_Clean"], sampling_rate=SAMPLE_RATE, show=True)

    # we can get the mean heart rate and a lot of HRV (heart rate variability) metrics
    ecg_info = nk.ecg_intervalrelated(data=ecg_df, sampling_rate=SAMPLE_RATE)

    # get heartbeats where the index of the dataframes equals the timestamp
    # the heartbeat length is different for every sample!
    ecg_heartbeats = nk.ecg_segment(ecg_cleaned=ecg_df["ECG_Clean"], sampling_rate=SAMPLE_RATE, show=True)
    plt.show()

    pass


def analysis_with_heartpy(sample):
    # resetting the index speeds up hp. plotter by a lot
    sample = sample.reset_index(drop=True)
    try:
        working_data, measures = hp.process(hrdata=sample, sample_rate=SAMPLE_RATE)
        hp.plotter(working_data, measures)

        # function to flip signal
        hrdata = hp.flip_signal(sample)
        working_data, measures = hp.process(hrdata=hrdata, sample_rate=SAMPLE_RATE)
        hp.plotter(working_data, measures)

        # enhances peak amplitude relative to rest of signal
        hrdata = hp.enhance_peaks(hrdata=sample)
        working_data, measures = hp.process(hrdata=hrdata, sample_rate=SAMPLE_RATE)
        hp.plotter(working_data, measures)

        # enhances ecg peaks by using synthetic QRS peaks
        hrdata = hp.enhance_ecg_peaks(hrdata=sample, sample_rate=SAMPLE_RATE)
        working_data, measures = hp.process(hrdata=hrdata, sample_rate=SAMPLE_RATE)
        hp.plotter(working_data, measures)
    except:
        print("Opps why ?!")


if __name__ == '__main__':
    # only read top n-rows for faster manual analysis
    n_rows = 20
    x_train = pd.read_csv("./data/X_train.csv", index_col=0, nrows=n_rows)
    y_train = pd.read_csv("./data/y_train.csv", index_col=0, nrows=n_rows)
    x_test = pd.read_csv("./data/X_test.csv", index_col=0, nrows=n_rows)

    warnings.filterwarnings("ignore")
    interactive = False
    if interactive:
        # pip install PyQt5
        import matplotlib as mpl

        mpl.use("Qt5Agg")

    for i in range(0, x_train.shape[0]):
        sample = x_train.iloc[i]

        # shorten series to non nan values
        last_non_nan_idx = pd.Series.last_valid_index(sample)
        sample = sample[:last_non_nan_idx]

        # Reset the index because it causes unwanted effects in the library functions!
        sample = sample.reset_index(drop=True)

        # get the label of the current sample
        label = y_train.iloc[i][0]
        print("\nclass:", label)

        analysis_with_biosppy_lib(sample)

        analysis_with_neurokit2(sample)

        # analysis_with_heartpy(sample)

        print("\n")

    pass
