"""
Main class
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import argparse
import os
import time
import xgboost as xgboost
import pandas as pd
import numpy as np
from helpers import argumenthelper
from logcreator.logcreator import Logcreator
from source.configuration import Configuration
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

def get_data(data, skip_first=0):
    feature_list = []

    feature_names = ["mean", "variance", "mean_heart_rate", "variance_heart_rate",
                     "max_hb_graph", "min_hb_graph", "perc10_hb_graph",
                     "perc25_hb_graph", "perc50_hb_graph", "perc75_hb_graph", "perc90_hb_graph",
                     "mean_rpeak_amp", "var_rpeak_amp", #first batch
                     "nni_mean", "nni_var", "biosppy_hrv",  #second batch
                     "qrs_complex_mean", "qrs_complex_var", "pr_interval_mean", "pr_interval_var",
                     "pr_segment_mean", "pr_segment_var", "qt_interval_mean", "pt_interval_var",
                     "st_segment_mean", "st_segment_var", "qrs_duration_mean", "qrs_duration_var",
                     "q_peak_amp_mean", "q_peak_amp_var" #third batch
                     ]

    for name in feature_names:
        if skip_first == 0:
            x_feature = pd.read_csv("./data/extracted_features/x_" + data + "_" + name + ".csv", index_col=0)
        else:
            x_feature = pd.read_csv(
                "./data/extracted_features/x_" + data + "_" + name + "_skip_first_" + str(skip_first) + ".csv",
                index_col=0)
        feature_list.append(x_feature)

    x_data = pd.concat(feature_list, axis=1)

    return x_data


if __name__ == "__main__":
    global config
    # Sample Config: --handin true --configuration D:\GitHub\AML\Task1\configurations\test.jsonc
    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/default.jsonc',
                        type=str, help="Environment and training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str,
                        help="Working directory (default: current directory).")
    parser.add_argument('--handin', default=False, type=argumenthelper.boolean_string,
                        help="If set to true, whole trainingset used for training")
    parser.add_argument('--hyperparamsearch', default=False, type=argumenthelper.boolean_string,
                        help="If set to true, will perform hyper parameter search, else it will only fit the given model")

    args = argumenthelper.parse_args(parser)
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir)
    Logcreator.initialize()

    Logcreator.h1("Task 03 - ECG Desease classification")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

    search = args.hyperparamsearch
    x_train_data = get_data("train", skip_first=300)
    x_test_handin = get_data("test", skip_first=300)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    handin_idx = x_test_handin.index

    # fill NaN with zero
    x_train_data = x_train_data.fillna(0)
    x_test_handin = x_test_handin.fillna(0)

    x_train_data[x_train_data == np.inf] = 0
    x_test_handin[x_test_handin == np.inf] = 0
    x_train_data[x_train_data == -np.inf] = 0
    x_test_handin[x_test_handin == -np.inf] = 0

    y_train_data = pd.read_csv("./data/y_train.csv", index_col=0)

    if not args.handin:
        x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data,
                                                        test_size=0.2,
                                                        stratify=y_train_data,
                                                        random_state=41)
    else:
        x_train = x_train_data
        y_train = y_train_data
        x_test = x_test_handin

    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = xgboost.XGBClassifier(objective='multi:softmax',
                                  n_estimators=897,
                                  max_depth=20,
                                  min_child_weight=3,
                                  reg_lambda=0,
                                  subsample=0.74,
                                  colsample_bytree=0.49,
                                  colsample_bylevel=0.64,
                                  num_class=3,
                                  n_jobs=-1,
                                  gamma=1,
                                  random_state=41)
    Logcreator.info("xgboost.XGBClassifier(objective='multi:softmax', "
                    "n_estimators=897, "
                    "max_depth=20, "
                    "min_child_weight=3, "
                    "reg_lambda=0, "
                    "subsample=0.74,"
                    "colsample_bytree=0.49,"
                    "colsample_bylevel=0.64,"
                    "num_class=3,"
                    "n_jobs=-1,"
                    "gamma=1,"
                    "random_state=41)")

    model.fit(x_train, y_train.values.flatten())

    # ---------------------------------------------------------------------------------------------------------
    # results
    y_pred_test = model.predict(x_test)

    if not args.handin:
        Logcreator.info("f1 score on test split", f1_score(y_true=y_test, y_pred=y_pred_test, average='micro'))

        Logcreator.info("\n", confusion_matrix(y_true=y_test, y_pred=y_pred_test))
    else:
        output_csv = pd.DataFrame(y_pred_test, columns=['y'], index=handin_idx)
        pd.DataFrame.to_csv(output_csv, os.path.join(Configuration.output_directory, "submission.csv"))

    end = time.time()
    Logcreator.info("Finished processing in %d [s]." % (end - start))
