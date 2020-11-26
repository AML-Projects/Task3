import warnings

import biosppy as biosppy
import heartpy as hp
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

# 300Hz sample rate
import wfdb
import numpy as np
from biosppy.signals import tools
from wfdb import processing

SAMPLE_RATE = 300


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    """"
    Plot a signal with its peaks and heart rate
    https://github.com/MIT-LCP/wfdb-python/blob/master/demo.ipynb
    """
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

    print(templates.shape)
    plt.plot(range(0, templates[0].shape[0]), templates[0])
    plt.show()
    if False:
        plt.plot(ts, filtered)
        plt.title("class " + str(label))
        plt.grid()
        plt.show()

    pass


def analysis_with_neurokit2(sample):
    # we have to reset the index, otherwise ecg_process returns unwanted results
    sample = sample.reset_index(drop=True)
    ecg_df, r_peaks = nk.ecg_process(ecg_signal=sample, sampling_rate=SAMPLE_RATE) #, method='engzeemod2012')
    #nk.ecg_plot(ecg_signals=ecg_df, sampling_rate=SAMPLE_RATE, show_type='full')
    #plt.show()

    print("neurokit2 rpeaks:", r_peaks)

    # ecg_cleaned = nk.ecg_clean(ecg_signal=sample, sampling_rate=SAMPLE_RATE, method='biosppy')
    # ecg_df, r_peaks = nk.ecg_process(ecg_signal=ecg_cleaned, sampling_rate=SAMPLE_RATE)
    # nk.ecg_plot(ecg_signals=ecg_df, sampling_rate=SAMPLE_RATE, show_type='full')
    # plt.show()

    # binary series describing where the peaks are
    ecg_df["ECG_P_Peaks"]
    ecg_df["ECG_Q_Peaks"]
    ecg_df["ECG_R_Peaks"]
    ecg_df["ECG_S_Peaks"]
    ecg_df["ECG_T_Peaks"]
    # plot
    nk.ecg_plot(ecg_signals=ecg_df, sampling_rate=SAMPLE_RATE) #, show_type='full')
    plt.show()

    # we can also just get the rpeaks
    r_peaks2 = nk.ecg_findpeaks(ecg_cleaned=ecg_df["ECG_Clean"], sampling_rate=SAMPLE_RATE, show=True)

    # we can get the mean heart rate and a lot of HRV (heart rate variability) metrics
    ecg_info = nk.ecg_intervalrelated(data=ecg_df, sampling_rate=SAMPLE_RATE)

    # get heartbeats where the index of the dataframes equals the timestamp
    # the heartbeat length is different for every sample!
    ecg_heartbeats = nk.ecg_segment(ecg_cleaned=ecg_df["ECG_Clean"], sampling_rate=SAMPLE_RATE, show=True)
    print("neurokit hb", len(ecg_heartbeats), ecg_heartbeats['1'].shape)
    plt.plot(range(0, ecg_heartbeats['1'].shape[0]), ecg_heartbeats['1']['Signal'].values)
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


def correct_r_peaks(sample, peaks):
    max_bpm = 230
    # Use the maximum possible bpm as the search radius
    search_radius = int(SAMPLE_RATE * 60 / max_bpm)
    try:
        corrected_peak_inds = processing.peaks.correct_peaks(sample,
                                                             peak_inds=peaks,
                                                             search_radius=search_radius,
                                                             smooth_window_size=150)

        if (corrected_peak_inds[0] < 0):  # sometimes the first index gets corrected to a negative value
            corrected_peak_inds = corrected_peak_inds[1:]

    except:
        corrected_peak_inds = peaks

    return corrected_peak_inds


def r_peak(sample):
    sampling_rate = float(SAMPLE_RATE)

    # filter signal
    order = int(0.3 * sampling_rate)
    sample_bio, _, _ = tools.filter_signal(signal=sample,
                                           ftype='FIR',
                                           band='bandpass',
                                           order=order,
                                           frequency=[3, 45],
                                           sampling_rate=sampling_rate)

    rpeaks_es = biosppy.signals.ecg.engzee_segmenter(sample_bio, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_es = correct_r_peaks(sample_bio, rpeaks_es)
    peaks_hr(sig=sample, peak_inds=rpeaks_es, fs=SAMPLE_RATE, title="R rpeaks_es")

    rpeaks_hs = biosppy.signals.ecg.hamilton_segmenter(sample_bio, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_hs = correct_r_peaks(sample_bio, rpeaks_hs)
    peaks_hr(sig=sample, peak_inds=rpeaks_hs, fs=SAMPLE_RATE, title="R rpeaks_hs")
    """
    rpeaks_cs = biosppy.ecg.christov_segmenter(sample, sampling_rate=SAMPLE_RATE)[0]
    rpeaks_cs = correct_r_peaks(sample, rpeaks_cs)
    peaks_hr(sig=sample, peak_inds=rpeaks_cs, fs=SAMPLE_RATE, title="R rpeaks_cs")
    """

    """
    # not so good
    rpeaks_gs = biosppy.ecg.gamboa_segmenter(sample, sampling_rate=SAMPLE_RATE, tol=0.02)[0]
    peaks_hr(sig=sample, peak_inds=rpeaks_gs, fs=SAMPLE_RATE, title="R rpeaks_gs")

    rpeaks_ssfs = biosppy.ecg.ssf_segmenter(sample, sampling_rate=SAMPLE_RATE, threshold=2000)[0]
    peaks_hr(sig=sample, peak_inds=rpeaks_ssfs, fs=SAMPLE_RATE, title="R rpeaks_ssfs")
    """

    xqrs = wfdb.processing.XQRS(sig=sample_bio, fs=SAMPLE_RATE)
    xqrs.detect(sampfrom=0, sampto='end', learn=True, verbose=0)
    qrs_inds = correct_r_peaks(sample_bio, xqrs.qrs_inds)
    peaks_hr(sig=sample, peak_inds=qrs_inds, fs=SAMPLE_RATE, title="R peaks wfdb")

    cleaned = nk.ecg_clean(ecg_signal=sample, sampling_rate=SAMPLE_RATE)

    neurokit = nk.ecg_findpeaks(nk.ecg_clean(sample, method="neurokit", sampling_rate=SAMPLE_RATE),
                                method="neurokit",
                                sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    neurokit = correct_r_peaks(sample, neurokit)
    peaks_hr(sig=sample, peak_inds=neurokit, fs=SAMPLE_RATE, title="R peaks neurokit")

    """"
    pantompkins1985 = nk.ecg_findpeaks(nk.ecg_clean(sample, method="pantompkins1985"),
                                       sampling_rate=SAMPLE_RATE,
                                       method="pantompkins1985")["ECG_R_Peaks"]
    pantompkins1985 = correct_r_peaks(sample, pantompkins1985)
    peaks_hr(sig=sample, peak_inds=pantompkins1985, fs=SAMPLE_RATE, title="R peaks pantompkins1985")
    """

    nabian2018 = nk.ecg_findpeaks(sample_bio, method="nabian2018", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    nabian2018 = correct_r_peaks(sample_bio, nabian2018)
    peaks_hr(sig=sample, peak_inds=nabian2018, fs=SAMPLE_RATE, title="R peaks nabian2018")

    """
    hamilton2002 = nk.ecg_findpeaks(nk.ecg_clean(sample, method="hamilton2002", sampling_rate=SAMPLE_RATE),
                                    method="hamilton2002", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    hamilton2002 = correct_r_peaks(sample, hamilton2002)
    peaks_hr(sig=sample, peak_inds=hamilton2002, fs=SAMPLE_RATE, title="R peaks hamilton2002")

    martinez2003 = nk.ecg_findpeaks(cleaned, method="martinez2003", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    martinez2003 = correct_r_peaks(sample, martinez2003)
    peaks_hr(sig=sample, peak_inds=martinez2003, fs=SAMPLE_RATE, title="R peaks martinez2003")

    christov2004 = nk.ecg_findpeaks(cleaned, method="christov2004", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    christov2004 = correct_r_peaks(sample, christov2004)
    peaks_hr(sig=sample, peak_inds=christov2004, fs=SAMPLE_RATE, title="R peaks christov2004")

    gamboa2008 = nk.ecg_findpeaks(nk.ecg_clean(sample, method="gamboa2008", sampling_rate=SAMPLE_RATE),
                                  method="gamboa2008", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    gamboa2008 = correct_r_peaks(sample, gamboa2008)
    peaks_hr(sig=sample, peak_inds=gamboa2008, fs=SAMPLE_RATE, title="R peaks gamboa2008")
    
    elgendi2010 = nk.ecg_findpeaks(nk.ecg_clean(sample, method="elgendi2010", sampling_rate=SAMPLE_RATE),
                                   method="elgendi2010", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    elgendi2010 = correct_r_peaks(sample, elgendi2010)
    peaks_hr(sig=sample, peak_inds=elgendi2010, fs=SAMPLE_RATE, title="R peaks elgendi2010")
    """

    """
    engzeemod2012 = nk.ecg_findpeaks(nk.ecg_clean(sample, method="engzeemod2012", sampling_rate=SAMPLE_RATE),
                                     method="engzeemod2012", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    engzeemod2012 = correct_r_peaks(sample, engzeemod2012)
    peaks_hr(sig=sample, peak_inds=engzeemod2012, fs=SAMPLE_RATE, title="R peaks engzeemod2012")
    """

    kalidas2017 = nk.ecg_findpeaks(nk.ecg_clean(sample, method="kalidas2017", sampling_rate=SAMPLE_RATE),
                                   method="kalidas2017", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"]
    kalidas2017 = correct_r_peaks(sample, kalidas2017)
    peaks_hr(sig=sample, peak_inds=kalidas2017, fs=SAMPLE_RATE, title="R peaks kalidas2017")

    """
    rodrigues2020 = np.asarray(
        nk.ecg_findpeaks(cleaned, method="rodrigues2020", sampling_rate=SAMPLE_RATE)["ECG_R_Peaks"])
    # rodrigues2020 = correct_r_peaks(sample, rodrigues2020)
    peaks_hr(sig=sample, peak_inds=rodrigues2020, fs=SAMPLE_RATE, title="R peaks rodrigues2020")

    promac = nk.ecg_findpeaks(sample, sampling_rate=SAMPLE_RATE, method="promac", show=False)["ECG_R_Peaks"]
    promac = correct_r_peaks(sample, promac)
    peaks_hr(sig=sample, peak_inds=promac, fs=SAMPLE_RATE, title="R peaks promac")
    """
    pass


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

        r_peak(sample)

        # analysis_with_heartpy(sample)

        print("\n")

    pass
