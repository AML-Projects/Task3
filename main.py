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
from sklearn.ensemble import RandomForestClassifier

from helpers import argumenthelper
from logcreator.logcreator import Logcreator
from source.configuration import Configuration
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler


def get_data(data, skip_first=0, features='111', folder=''):
    """

    :param data: train / test
    :param skip_first:
    :param features: 111 = include all features, 110 = include features 1 and 2, ...
    :return:
    """
    feature_list = []
    feature_names = []

    feature_names_1 = ["mean", "variance", "mean_heart_rate", "variance_heart_rate",
                       "max_hb_graph", "min_hb_graph", "perc10_hb_graph",
                       "perc25_hb_graph", "perc50_hb_graph", "perc75_hb_graph", "perc90_hb_graph"
                       ]

    feature_names_2 = ["nni_mean", "nni_var", "biosppy_hrv"]  # second batch

    feature_names_3 = ["qrs_complex_mean", "qrs_complex_var", "pr_interval_mean", "pr_interval_var",
                       "pr_segment_mean", "pr_segment_var", "qt_interval_mean", "pt_interval_var",
                       "st_segment_mean", "st_segment_var", "qrs_duration_mean", "qrs_duration_var",
                       "q_peak_amp_mean", "q_peak_amp_var"]  # third batch

    # feature_names_3 = [x + '_filtered' for x in feature_names_3]

    if features[0] == '1':
        feature_names.extend(feature_names_1)
    if features[1] == '1':
        feature_names.extend(feature_names_2)
    if features[2] == '1':
        feature_names.extend(feature_names_3)

    for name in feature_names:
        path_to_files = "./data/extracted_features/"
        if folder != '':
            path_to_files = path_to_files + folder + "/"
        if skip_first == 0:
            x_feature = pd.read_csv(path_to_files + "x_" + data + "_" + name + ".csv", index_col=0)
        else:
            x_feature = pd.read_csv(path_to_files + "/x_" + data + "_" + name + "_skip_first_" + str(skip_first) + ".csv", index_col=0)
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
    parser.add_argument('--cvscore', default=True, type=argumenthelper.boolean_string,
                        help="If True does perform cross validation on the training set.")

    args = argumenthelper.parse_args(parser)
    start = time.time()

    Configuration.initialize(args.configuration, args.workingdir)
    Logcreator.initialize()

    Logcreator.h1("Task 03 - ECG Desease classification")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

    search = args.hyperparamsearch
    x_train_data = get_data("train", skip_first=300, features='111')
    x_test_handin = get_data("test", skip_first=300, features='111')

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

    parameter = {
        'objective': 'multi:softmax',
        'n_estimators': 897,
        'max_depth': 20,

        'min_child_weight': 3,
        'reg_lambda': 0,
        'subsample': 0.74,
        'colsample_bytree': 0.49,
        'colsample_bylevel': 0.64,
        'gamma': 1,

        'random_state': 41
    }

    model = xgboost.XGBClassifier(**parameter)
    # model = RandomForestClassifier(n_estimators=500, random_state=41)

    Logcreator.info(model)

    if args.cvscore:
        Logcreator.info("Running CV on the training set")
        sk = StratifiedKFold(shuffle=True, n_splits=10, random_state=41)
        Logcreator.info("CV-score", cross_val_score(model, x_train, y_train.values.flatten(), cv=sk, scoring='f1_micro', n_jobs=-1).mean())

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
