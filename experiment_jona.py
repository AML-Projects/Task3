import numpy as np
import pandas as pd
import xgboost as xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler

from logcreator.logcreator import Logcreator


def get_data(data, skip_first=0):
    feature_list = []

    feature_names = ["mean", "variance", "mean_heart_rate", "variance_heart_rate",
                     "max_hb_graph", "min_hb_graph",
                     "perc25_hb_graph", "perc50_hb_graph", "perc75_hb_graph",
                     "nni_mean", "nni_var", "biosppy_hrv"]  # , "diff_mean"]

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


if __name__ == '__main__':
    search = False
    x_train_data = get_data("train")
    x_test_handin = get_data("test")

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

    x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data,
                                                        test_size=0.2,
                                                        stratify=y_train_data,
                                                        random_state=41)

    scaler = RobustScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_test_handin = scaler.transform(x_test_handin)

    model = xgboost.XGBClassifier(objective='multi:softmax',
                                  n_estimators=100,
                                  max_depth=8,
                                  subsample=1,
                                  colsample_bytree=1,
                                  colsample_bylevel=0.6,
                                  num_class=3,
                                  n_jobs=-1,
                                  random_state=41)

    params = {
        'objective': ['multi:softmax'],
        # setting max_depth to high results in overfitting
        # 'max_depth': [4, 6, 8],
        # subsampling of rows: lower values of subsample can prevent overfitting
        # 'subsample': [0.9, 1],
        # 'colsample_bytree': [0.6, 0.8, 1],
        # 'colsample_bylevel': [0.6, 0.8, 1]
    }

    """
    model = RandomForestClassifier(random_state=41)
    params = {
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", None],
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    }
    """

    if search:
        sk = StratifiedKFold(shuffle=True, n_splits=10, random_state=41)
        clf = RandomizedSearchCV(model, params,
                                 n_iter=100,
                                 scoring="f1_micro",
                                 refit=True,
                                 cv=sk,
                                 return_train_score=True,
                                 verbose=1,
                                 n_jobs=-1,
                                 random_state=41)

        Logcreator.info("Best estimator from GridSearch: {}".format(clf.best_estimator_))
        Logcreator.info("Best alpha found: {}".format(clf.best_params_))
        Logcreator.info("Best training-score with mse loss: {}".format(clf.best_score_))

        results = pd.DataFrame(clf.cv_results_)
        results.sort_values(by=['rank_test_score'], inplace=True)
        col = [c for c in results.columns if 'split' not in c and 'time' not in c and 'params' not in c]
        Logcreator.info(results[col])

    else:
        clf = model

    clf.fit(x_train, y_train.values.flatten())

    # ---------------------------------------------------------------------------------------------------------
    # results
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)

    Logcreator.info("f1 score on train split", f1_score(y_true=y_train, y_pred=y_pred_train, average='micro'))
    Logcreator.info("f1 score on test split", f1_score(y_true=y_test, y_pred=y_pred_test, average='micro'))

    Logcreator.info("\n", confusion_matrix(y_true=y_train, y_pred=y_pred_train))
    Logcreator.info("\n", confusion_matrix(y_true=y_test, y_pred=y_pred_test))

    # ---------------------------------------------------------------------------------------------------------
    # refit on everything
    clf.fit(x_train_data, y_train_data)
    y_pred_test_handin = clf.predict(x_test_handin)
    pd.DataFrame(y_pred_test_handin, columns=['y'], index=handin_idx).to_csv("./trainings/submission.csv")
