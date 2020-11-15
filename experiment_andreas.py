




import argparse
import os
import time
import math
import pandas as pd
from sklearn import model_selection

from helpers import argumenthelper
from logcreator.logcreator import Logcreator
from source.configuration import Configuration
from matplotlib import pyplot
#from biosppy import storage
from biosppy.signals import ecg


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
    SAMPLE_RATE = 300
    # Load training data
    x_train = pd.read_csv("./data/X_train.csv", index_col=0, nrows=20)
    y_train = pd.read_csv("./data/y_train.csv", index_col=0, nrows=20)
    x_test = pd.read_csv("./data/X_test.csv", index_col=0, nrows=20)

    x_train = x_train.T.reset_index(drop=True).T
    x_train.reset_index(drop=True, inplace=True)
    y_train = y_train.T.reset_index(drop=True).T
    y_train.reset_index(drop=True, inplace=True)
    x_test = x_test.T.reset_index(drop=True).T
    x_test.reset_index(drop=True, inplace=True)

    print("x_train:")
    print(x_train.head())
    print("y_train:")
    print(y_train.head())
    print("x_test:")
    print(x_test.head())

    #Print the first 5 rows
    ploted_lines = pd.DataFrame()
    for index in range(0, 4):
        ploted_lines[index] = x_train.values[index, 250:]
        name = "Line " + str(index) + ": Class " + str(y_train.values[index, 0])
        ploted_lines = ploted_lines.rename(columns={index : name})
        #last_valid = pd.Series.last_valid_index(ploted_lines[name])
        #signal = ploted_lines[name].values[100:last_valid+1]
        #ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr = ecg.ecg(signal=signal, sampling_rate=SAMPLE_RATE,
        #                                                          show=True)


    x_train_chopped = pd.DataFrame()
    y_train_chopped = pd.DataFrame()
    #Sample-chopping:
    for index, row in x_train.iterrows():
        last_valid = pd.Series.last_valid_index(row)
        row = row[250:last_valid+1] #First we cut the first 250 samples 0.83s and also cut the nan-values at the end

        blocksize = SAMPLE_RATE*5
        nr_of_blocks = math.floor(len(row)/(blocksize))
        for block_nr in range(nr_of_blocks):
            begin = block_nr*blocksize
            end = (block_nr+1)*blocksize
            sample_block = row[begin:end].reset_index(drop=True)
            x_train_chopped = x_train_chopped.append(sample_block, ignore_index=True)
            y_train_chopped = y_train_chopped.append(pd.Series(y_train.iloc[index, 0]), ignore_index=True)

        print(row)
        print(index)


    ploted_lines.plot(subplots=True, legend=True)
    pyplot.show()
    x_train_chopped.iloc[0:5, ].T.plot(subplots=True, legend=True)
    pyplot.show()


    #whateverthisis= ecg.extract_heartbeats(signal=ploted_lines.values[:, 0], rpeaks=None, sampling_rate=SAMPLE_RATE)






    print ("finish")