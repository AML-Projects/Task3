import pandas as pd
import xgboost as xgboost
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from logcreator.logcreator import Logcreator


def get_data(data):
    feature_list = []

    feature_names = ["mean", "variance", "mean_heart_rate", "variance_heart_rate",
                     "max_hb_graph", "min_hb_graph",
                     "perc25_hb_graph", "perc50_hb_graph", "perc75_hb_graph"]  # , "diff_mean"]

    for name in feature_names:
        x_feature = pd.read_csv("./data/extracted_features/x_" + data + "_" + name + ".csv", index_col=0)
        feature_list.append(x_feature)

    x_data = pd.concat(feature_list, axis=1)

    return x_data


if __name__ == '__main__':
    x_train_data = get_data("train")
    x_test_handin = get_data("test")

    handin_idx = x_test_handin.index

    # fill NaN with zero
    x_train_data = x_train_data.fillna(0)
    x_test_handin = x_test_handin.fillna(0)

    y_train_data = pd.read_csv("./data/y_train.csv", index_col=0)

    x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data,
                                                        test_size=0.2,
                                                        stratify=y_train_data,
                                                        random_state=41)

    scaler = RobustScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_test_handin = scaler.transform(x_test_handin)

    clf = xgboost.XGBClassifier(objective='multi:softmax',
                                n_estimators=100,
                                max_depth=6,
                                subsample=1,
                                colsample_bytree=1,
                                colsample_bylevel=1,
                                num_class=3,
                                n_jobs=-1,
                                random_state=41)

    clf.fit(x_train, y_train)

    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    y_pred_test_handin = clf.predict(x_test_handin)

    Logcreator.info("f1 score on train split", f1_score(y_true=y_train, y_pred=y_pred_train, average='micro'))
    Logcreator.info("f1 score on test split", f1_score(y_true=y_test, y_pred=y_pred_test, average='micro'))

    pd.DataFrame(y_pred_test_handin, columns=['y'], index=handin_idx).to_csv("./trainings/submission.csv")
