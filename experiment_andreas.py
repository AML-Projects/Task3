




import argparse
import os
import time

import pandas as pd
from sklearn import model_selection

from helpers import argumenthelper
from logcreator.logcreator import Logcreator
from source.configuration import Configuration

if __name__ == "__main__":
    global config
    # Sample Config: --handin true --configuration D:\GitHub\AML\Task1\configurations\test.jsonc
    parser = argparse.ArgumentParser(
        description="Executes a training session.")
    parser.add_argument('--configuration', default='./configurations/e1.jsonc',
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

    # Load training data
    x_train = pd.read_csv("./data/X_train.csv", index_col=0)
    y_train = pd.read_csv("./data/y_train.csv", index_col=0)
    x_test = pd.read_csv("./data/X_test.csv", index_col=0)

    x_train



