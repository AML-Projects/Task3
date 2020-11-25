import numpy as np
import optuna
import pandas as pd
import plotly  # for optuna plot
import xgboost as xgboost
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_slice, plot_parallel_coordinate, plot_contour
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier


def get_data(data, skip_first=0):
    feature_list = []

    feature_names = ["mean", "variance",
                     "mean_heart_rate", "variance_heart_rate",
                     "max_hb_graph", "min_hb_graph",
                     "perc10_hb_graph", "perc25_hb_graph", "perc50_hb_graph", "perc75_hb_graph", "perc90_hb_graph",
                     "nni_mean", "nni_var",
                     "biosppy_hrv"]

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


def objective_random_forest(trial: Trial, X, y) -> float:
    params = {
        'criterion': trial.suggest_categorical('criterion', ["gini", "entropy"]),
        'class_weight': trial.suggest_categorical('class_weight', ["balanced", None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),

        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int("max_depth", 2, 32),
        'max_features': trial.suggest_uniform('max_features', 0.15, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 16),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 16),
        'max_samples': trial.suggest_uniform('max_samples', 0.6, 1),
    }

    model = RandomForestClassifier(**params,
                                   n_jobs=-1,
                                   random_state=41)

    sk = StratifiedKFold(shuffle=True, n_splits=10, random_state=41)
    return cross_val_score(model, X, y, cv=sk, scoring='f1_micro', n_jobs=-1).mean()


def objective_xgb(trial: Trial, X, y) -> float:
    param = {
        "n_estimators": trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 25),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
        'gamma': trial.suggest_int('gamma', 0, 5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.01),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
        'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.1, 1, 0.01),
        'nthread': -1,
        'random_state': 41,
        'num_class': 3,
        'use_label_encoder': False,
    }

    model = XGBClassifier(objective='multi:softmax', **param)

    sk = StratifiedKFold(shuffle=True, n_splits=10, random_state=41)
    return cross_val_score(model, X, y, cv=sk, scoring='f1_micro', n_jobs=-1).mean()


if __name__ == '__main__':
    search = True
    plot_on = False
    study_model = "xgboost"
    study_model = "forest"

    # ---------------------------------------------------------------------------------------------------------
    x_train_data = get_data("train", 300)
    x_test_handin = get_data("test", 300)

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

    study = optuna.create_study(study_name=study_model,
                                direction='maximize',
                                storage='sqlite:///trainings/search.db',
                                # n_startup_trials: number of random searches before TPE starts
                                sampler=TPESampler(n_startup_trials=20),
                                load_if_exists=True)

    if search:
        print('Starting search')
        if study_model == "xgboost":
            study.optimize(lambda trial: objective_xgb(trial, x_train, y_train.values.flatten()),
                           # number of different parameter combinations to try out
                           n_trials=100,
                           n_jobs=-1)  # set to -3 to have 1 cpu free of load

        elif study_model == "forest":
            study.optimize(lambda trial: objective_random_forest(trial, x_train, y_train.values.flatten()),
                           n_trials=100,
                           n_jobs=-1)

        else:
            raise ValueError("study_model", study_model, "does not exist!")

        print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))
        hist = study.trials_dataframe()
        hist.to_csv("./trainings/study_long_search.csv")

        if plot_on:
            plot_optimization_history(study).show()
            plot_slice(study).show()
            plot_parallel_coordinate(study).show()
            plot_contour(study).show()

    print('Selecting best found model')
    if study_model == "xgboost":
        clf = xgboost.XGBClassifier(objective='multi:softmax', **study.best_trial.params,
                                    num_class=3,
                                    nthread=-1,
                                    random_state=41)
    elif study_model == "forest":
        clf = RandomForestClassifier(**study.best_trial.params,
                                     n_jobs=-1,
                                     random_state=41)
    else:
        raise ValueError("study_model", study_model, "does not exist!")

    print('Fitting model on training set')
    clf.fit(x_train, y_train.values.flatten())

    # ---------------------------------------------------------------------------------------------------------
    # results
    print('Predicting on training and test set')
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)

    print("f1 score on train split", f1_score(y_true=y_train, y_pred=y_pred_train, average='micro'))
    print("f1 score on test split", f1_score(y_true=y_test, y_pred=y_pred_test, average='micro'))

    print("\n", confusion_matrix(y_true=y_train, y_pred=y_pred_train))
    print("\n", confusion_matrix(y_true=y_test, y_pred=y_pred_test))

    # ---------------------------------------------------------------------------------------------------------
    # submission: refit on everything
    print('Fitting model on all training data')
    x_train_data = scaler.fit_transform(x_train_data)
    x_test_handin = scaler.transform(x_test_handin)

    clf.fit(x_train_data, y_train_data.values.flatten())

    y_pred_test_handin = clf.predict(x_test_handin)

    pd.DataFrame(y_pred_test_handin, columns=['y'], index=handin_idx).to_csv("./trainings/submission.csv")
