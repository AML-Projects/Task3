import os
import time

import biosppy as biosppy
import neurokit2 as nk2
import numpy as np
import pandas as pd
import pyhrv
import wfdb
from matplotlib import pyplot as plt
from pyhrv import tools
from wfdb import processing

from logcreator.logcreator import Logcreator

# 300Hz sample rate
SAMPLE_RATE = 300


class RPeakDetector:
    r_peak_detection_method = 'biosppy'

    @staticmethod
    def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
        """"
        Plot a signal with its peaks and heart rate
        https://github.com/MIT-LCP/wfdb-python/blob/master/demo.ipynb
        """
        fig_length = peak_inds.shape[0]
        if fig_length < 10:
            fig_length = 10
        figsize = (fig_length, 12)

        # Calculate heart rate
        hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)

        N = sig.shape[0]

        fig, ax_left = plt.subplots(figsize=figsize)
        ax_right = ax_left.twinx()

        ax_left.plot(sig, color='#3979f0', label='Signal')
        ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x',
                     color='#8b0000', label='Peak', markersize=12)
        ax_right.plot(np.arange(N), hrs, label='Heart rate', color='m', linewidth=2)

        ax_left.set_title(title)

        ax_left.set_xlabel('Time (ms)')
        ax_left.set_ylabel('ECG (mV)', color='#3979f0')
        ax_right.set_ylabel('Heart rate (bpm)', color='m')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax_left.tick_params('y', colors='#3979f0')
        ax_right.tick_params('y', colors='m')
        if saveto is not None:
            plt.savefig(saveto, dpi=600)
        plt.show()
        pass

    @staticmethod
    def correct_r_peaks(sample, peaks):
        max_bpm = 200
        # Use the maximum possible bpm as the search radius
        search_radius = int(SAMPLE_RATE * 60 / max_bpm)
        try:
            corrected_peak_inds = processing.peaks.correct_peaks(sample,
                                                                 peak_inds=peaks,
                                                                 search_radius=search_radius,
                                                                 smooth_window_size=150)

            if corrected_peak_inds[0] < 0:  # sometimes the first index gets corrected to a negative value
                corrected_peak_inds = corrected_peak_inds[1:]

        except:
            corrected_peak_inds = peaks

        return corrected_peak_inds

    @staticmethod
    def get_r_peaks(sample, library=None):
        """
        :param sample: Supply unfiltered signal, does filtering!
        :param library:
        :return:
        """
        if library is None:
            library = RPeakDetector.r_peak_detection_method

        if library == 'biosppy':
            # filter signal
            filtered = filter_signal(sample)
            # segment
            r_peaks, = biosppy.signals.ecg.hamilton_segmenter(signal=filtered, sampling_rate=SAMPLE_RATE)
            # correct R-peak locations
            r_peaks, = biosppy.signals.ecg.correct_rpeaks(signal=filtered,
                                                          rpeaks=r_peaks,
                                                          sampling_rate=SAMPLE_RATE,
                                                          tol=0.05)

            # extract templates -> this function corrects r-peaks further!
            templates, r_peaks = biosppy.signals.ecg.extract_heartbeats(signal=filtered,
                                                                        rpeaks=r_peaks,
                                                                        sampling_rate=SAMPLE_RATE,
                                                                        before=0.2,
                                                                        after=0.4)
        elif library == 'neurokit':
            method = 'neurokit'  # "kalidas2017"
            # filter signal
            filtered = nk2.ecg_clean(sample, method=method, sampling_rate=SAMPLE_RATE)
            # find peaks
            r_peaks = nk2.ecg_findpeaks(ecg_cleaned=filtered, method=method, sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]

        elif library == 'wfdb':
            # filter signal
            filtered = filter_signal(sample)

            xqrs = wfdb.processing.XQRS(sig=filtered, fs=SAMPLE_RATE)
            xqrs.detect(sampfrom=0, sampto='end', learn=True, verbose=0)
            r_peaks = xqrs.qrs_inds
            # Plot results
            # peaks_hr(sig=sample, peak_inds=qrs_inds, fs=SAMPLE_RATE, title="R peaks")

            if r_peaks.shape[0] == 0:
                return np.ndarray([])

            r_peaks = RPeakDetector.correct_r_peaks(filtered, r_peaks)

            # remove first r-peak: by manual analysis the first one is often incorrect
            r_peaks = r_peaks[1:]

            # extract templates -> this function corrects r-peaks further!
            templates, r_peaks = biosppy.signals.ecg.extract_heartbeats(signal=filtered,
                                                                        rpeaks=r_peaks,
                                                                        sampling_rate=SAMPLE_RATE,
                                                                        before=0.2,
                                                                        after=0.4)

        else:
            raise AssertionError("Library does not exist:", library)

        # sometimes two peaks are the same
        r_peaks = np.unique(r_peaks)

        return r_peaks


def filter_signal(sample):
    """
    If applied to an already filtered signal, filters the signal even further!
    :param sample:
    :return:
    """

    """
    # filter signal with biosppy
    order = int(0.3 * SAMPLE_RATE)
    sample_filtered, _, _ = biosppy.tools.filter_signal(signal=sample,
                                                        ftype='FIR',
                                                        band='bandpass',
                                                        order=order,
                                                        frequency=[3, 45],
                                                        sampling_rate=SAMPLE_RATE)
    """
    # this one line of code does the same as the commented code above
    sample_filtered = nk2.ecg_clean(ecg_signal=sample, sampling_rate=SAMPLE_RATE, method='biosppy')

    return sample_filtered


def extract_mean_variance(sample, show=False):
    """
    Extracts the mean and variance heart beat, plus a lot more of a given sample.

    :return: mean_hb_graph, var_hb_graph,
             mean_hb_rate, var_hb_rate,
             max_hb_graph, min_hb_graph,
             perc25_hb_graph, perc50_hb_graph, perc75_hb_graph
             diff_mean
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
    var_hb_rate = np.var(heart_rate, axis=0)
    max_hb_graph = np.amax(heartbeat_templates, axis=0)
    perc10_hb_graph = np.percentile(heartbeat_templates, q=10, axis=0)
    perc25_hb_graph = np.percentile(heartbeat_templates, q=25, axis=0)
    perc50_hb_graph = np.percentile(heartbeat_templates, q=50, axis=0)
    perc75_hb_graph = np.percentile(heartbeat_templates, q=75, axis=0)
    perc90_hb_graph = np.percentile(heartbeat_templates, q=90, axis=0)
    min_hb_graph = np.min(heartbeat_templates, axis=0)
    mean_rpeak_amp = np.mean(sample[rpeaks,], axis=0)
    var_rpeak_amp = np.var(sample[rpeaks,], axis=0)

    if show:
        plt.plot(range(0, perc25_hb_graph.shape[0]), max_hb_graph)
        plt.plot(range(0, perc25_hb_graph.shape[0]), min_hb_graph)
        plt.plot(range(0, perc25_hb_graph.shape[0]), perc25_hb_graph)
        plt.plot(range(0, perc50_hb_graph.shape[0]), perc50_hb_graph)
        plt.plot(range(0, perc75_hb_graph.shape[0]), perc75_hb_graph)
        plt.show()

    return mean_hb_graph, var_hb_graph, \
           mean_hb_rate, var_hb_rate, \
           max_hb_graph, min_hb_graph, \
           perc10_hb_graph, perc25_hb_graph, perc50_hb_graph, perc75_hb_graph, perc90_hb_graph, \
           mean_rpeak_amp, var_rpeak_amp


def extract_nni(sample, r_peaks=None):
    if r_peaks is None:
        r_peaks = RPeakDetector.get_r_peaks(sample)

    if r_peaks.shape[0] == 0:
        return 0, 0
    elif r_peaks.shape[0] == 1:
        nni = r_peaks
    else:
        nni = pyhrv.tools.nn_intervals(rpeaks=r_peaks)

    nni_mean = np.mean(nni, axis=0)
    nni_var = np.var(nni, axis=0)

    return nni_mean, nni_var


def extract_hrv_nk2(sample, r_peaks=None):
    try:
        if r_peaks is None:
            r_peaks = RPeakDetector.get_r_peaks(sample)

        r_peaks_dict = {"ECG_R_Peaks": r_peaks}

        ecg_info = nk2.hrv(peaks=r_peaks_dict, sampling_rate=SAMPLE_RATE, show=False)

    except (ValueError, IndexError) as e:
        # probably not enough r_peaks or probably class 3
        print("ValueError:", str(e))
        # nk2.ecg_findpeaks(ecg_cleaned=sample, sampling_rate=SAMPLE_RATE, show=True)
        # plt.show()
        RPeakDetector.peaks_hr(sample, r_peaks, SAMPLE_RATE, title="failed hrv")

        return [np.zeros(52)]

    return [ecg_info.values.flatten()]


def extract_hrv_and_nni(sample):
    r_peaks = RPeakDetector.get_r_peaks(sample)
    res1, res2 = extract_nni(sample, r_peaks)
    res3 = extract_hrv_nk2(sample, r_peaks)

    return res1, res2, np.asarray(res3).flatten()


def get_qrspt_features(r_peaks, sample):
    signal_dwt, waves_dwt = nk2.ecg_delineate(sample, r_peaks, sampling_rate=SAMPLE_RATE, method="dwt", show=False)
    signal_peak, waves_peak = nk2.ecg_delineate(sample, r_peaks, sampling_rate=SAMPLE_RATE, method="peak", show=False)

    t_peaks = np.array(waves_dwt['ECG_T_Peaks'])
    t_onsets = np.array(waves_dwt['ECG_T_Onsets'])
    t_offsets = np.array(waves_dwt['ECG_T_Offsets'])
    p_peaks = np.array(waves_dwt['ECG_P_Peaks'])
    p_onsets = np.array(waves_dwt['ECG_P_Onsets'])
    p_offsets = np.array(waves_dwt['ECG_P_Offsets'])
    r_onsets = np.array(waves_dwt['ECG_R_Onsets'])
    r_offsets = np.array(waves_dwt['ECG_R_Offsets'])
    q_peak = np.array(waves_peak['ECG_Q_Peaks'])
    s_peak = np.array(waves_peak['ECG_S_Peaks'])

    qrs_complex = r_offsets - r_onsets
    pr_interval = r_onsets - p_onsets
    pr_segment = r_onsets - p_offsets
    qt_interval = t_offsets - r_onsets
    st_segment = t_onsets - r_offsets
    qrs_duration = s_peak - q_peak

    qrs_complex_mean = np.mean(qrs_complex[~np.isnan(qrs_complex)])
    qrs_complex_var = np.var(qrs_complex[~np.isnan(qrs_complex)])
    pr_interval_mean = np.mean(pr_interval[~np.isnan(pr_interval)])
    pr_interval_var = np.var(pr_interval[~np.isnan(pr_interval)])
    pr_segment_mean = np.mean(pr_segment[~np.isnan(pr_segment)])
    pr_segment_var = np.var(pr_segment[~np.isnan(pr_segment)])
    qt_interval_mean = np.mean(qt_interval[~np.isnan(qt_interval)])
    pt_interval_var = np.var(qt_interval[~np.isnan(qt_interval)])
    st_segment_mean = np.mean(st_segment[~np.isnan(st_segment)])
    st_segment_var = np.var(st_segment[~np.isnan(st_segment)])
    qrs_duration_mean = np.mean(qrs_duration[~np.isnan(qrs_duration)])
    qrs_duration_var = np.var(qrs_duration[~np.isnan(qrs_duration)])
    q_peak_amp_mean = np.mean(sample[q_peak[~np.isnan(q_peak)].astype(int),], axis=0)
    q_peak_amp_var = np.var(sample[q_peak[~np.isnan(q_peak)].astype(int),], axis=0)

    return pr_interval_mean, pr_interval_var, pr_segment_mean, pr_segment_var, pt_interval_var, q_peak_amp_mean, q_peak_amp_var, \
           qrs_complex_mean, qrs_complex_var, qrs_duration_mean, qrs_duration_var, qt_interval_mean, st_segment_mean, st_segment_var


def exctract_qrspt(sample):
    r_peaks = RPeakDetector.get_r_peaks(sample)
    # filter signal first
    filtered = filter_signal(sample)
    try:
        pr_interval_mean, pr_interval_var, pr_segment_mean, pr_segment_var, pt_interval_var, \
        q_peak_amp_mean, q_peak_amp_var, qrs_complex_mean, qrs_complex_var, qrs_duration_mean, \
        qrs_duration_var, qt_interval_mean, st_segment_mean, st_segment_var = get_qrspt_features(r_peaks, filtered)
    except:
        RPeakDetector.peaks_hr(sample, r_peaks, SAMPLE_RATE, title="Fail1: qrspt extraction")
        # fall back to biosppy if it is not already biosppy
        if RPeakDetector.r_peak_detection_method == "biosppy":
            print("Extract qrspt found invalid sample")
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        r_peaks = RPeakDetector.get_r_peaks(sample, 'biosppy')
        RPeakDetector.peaks_hr(sample, r_peaks, SAMPLE_RATE, title="biosppy r-peaks")
        try:
            pr_interval_mean, pr_interval_var, pr_segment_mean, pr_segment_var, pt_interval_var, \
            q_peak_amp_mean, q_peak_amp_var, qrs_complex_mean, qrs_complex_var, qrs_duration_mean, \
            qrs_duration_var, qt_interval_mean, st_segment_mean, st_segment_var = get_qrspt_features(r_peaks, filtered)

        except:
            print("Extract qrspt found invalid sample")
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    return qrs_complex_mean, qrs_complex_var, pr_interval_mean, pr_interval_var, \
           pr_segment_mean, pr_segment_var, qt_interval_mean, pt_interval_var, st_segment_mean, st_segment_var, \
           qrs_duration_mean, qrs_duration_var, q_peak_amp_mean, q_peak_amp_var


def extract_frequency_domain(sample):
    from scipy.fft import rfft, rfftfreq, dct
    # fixed number of sample points
    N = 300
    # sample spacing
    T = 1.0 / SAMPLE_RATE
    # Flourier transformation
    y_rfft = rfft(sample, N)
    # x_frequency = rfftfreq(N, T)

    x_frequency = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    y_rfft = 2.0 / N * np.abs(y_rfft[0:N // 2])

    # to int -> filters the peaks a little bit and saves space
    y_rfft = y_rfft.astype(int)

    # cut off frequencies above 75Hz
    x_frequency = x_frequency[:len(x_frequency) // 2]
    y_rfft = y_rfft[:len(y_rfft) // 2]

    """
    plt.vlines(x_frequency, np.zeros(len(y_rfft)), abs(y_rfft))
    plt.grid()
    plt.show()
    """

    return [y_rfft]


def extract_features(x, x_name, extract_function, extracted_column_names, skip_first=0, skip_last=300):
    """
    General function to extract features and save them individually to a file.

    :param x: input data
    :param x_name: the name of the input data
    :param extract_function: function that extracts features for one sample of x
    :param extracted_column_names: the names of the extracted features
    :param skip_last: skips the first n data points in every sample
    :param skip_first: skips the last n data points in every sample
    :return:
    """
    start_time = time.time()
    Logcreator.info("\n", x_name, "start")

    # save index for later when saving the features
    index = x.index

    # Reset the index because it causes unwanted effects in the library functions!
    x = x.reset_index(drop=True)

    feature_list = []
    value_error_count = 0
    for i in range(0, x.shape[0]):
        if (i % int(x.shape[0] / 10)) == 0:
            elapsed_time = time.time() - start_time
            print(int(i / int(x.shape[0]) * 100), "% - samples processed:", i, "in %d [s]." % elapsed_time)

        sample = x.iloc[i]

        # shorten series to non nan values
        last_non_nan_idx = pd.Series.last_valid_index(sample)
        sample = sample[:last_non_nan_idx]

        # sample from dataframe to ndarray
        sample = sample.values

        # skip first and last n data points
        if skip_last == 0:
            sample = sample[skip_first:]
        else:
            sample = sample[skip_first:-skip_last]

        extracted_values = extract_function(sample)
        feature_list.append(extracted_values)

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
        if skip_first > 0:
            file_name = x_name + "_" + str(column.name) + "_skip_first_" + str(skip_first) + ".csv"
        else:
            file_name = x_name + "_" + str(column.name) + ".csv"
        if not os.path.exists(r"./data/extracted_features"):
            os.makedirs(r"./data/extracted_features")
        # save files in a different directory depending on the r-peak detection method
        directory = r"./data/extracted_features/" + RPeakDetector.r_peak_detection_method
        if not os.path.exists(directory):
            os.makedirs(directory)
        extracted.to_csv(os.path.join(directory, file_name), index=True)

    total_elapsed_time = time.time() - start_time
    Logcreator.info("\nFeature extraction finished in %d [s]." % total_elapsed_time)


if __name__ == '__main__':
    # set n_rows to a integer for testing, to read only the top n-rows
    n_rows = None
    # set r peak detection
    RPeakDetector.r_peak_detection_method = 'biosppy'  # biosppy, wfdb
    # set number of data points to skip
    skip_first = 300
    skip_last = 300

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

    extracted_feature_names = ["frequency_domain"]
    extract_features(x_train, "x_train", extract_frequency_domain, extracted_feature_names, skip_first=0, skip_last=0)
    extract_features(x_test, "x_test", extract_frequency_domain, extracted_feature_names, skip_first=0, skip_last=0)

    extracted_feature_names = ["nni_mean", "nni_var", "biosppy_hrv"]
    extract_features(x_train, "x_train", extract_hrv_and_nni, extracted_feature_names,
                     skip_first=skip_first,
                     skip_last=skip_last)
    extract_features(x_test, "x_test", extract_hrv_and_nni, extracted_feature_names,
                     skip_first=skip_first,
                     skip_last=skip_last)
    # exit()

    # extracting mean, variance and mean heart rate
    extracted_feature_names = ["mean", "variance",
                               "mean_heart_rate", "variance_heart_rate",
                               "max_hb_graph", "min_hb_graph",
                               "perc10_hb_graph", "perc25_hb_graph", "perc50_hb_graph", "perc75_hb_graph",
                               "perc90_hb_graph", "mean_rpeak_amp", "var_rpeak_amp"]
    # took 387s
    extract_features(x_train, "x_train", extract_mean_variance, extracted_feature_names,
                     skip_first=skip_first,
                     skip_last=skip_last)
    # took 256s
    extract_features(x_test, "x_test", extract_mean_variance, extracted_feature_names,
                     skip_first=skip_first,
                     skip_last=skip_last)
