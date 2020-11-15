"""
Initializes config, loads configfiles, adds folder for run-logs
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import commentjson
import os

import datetime
from shutil import copy

from helpers import converter
from helpers import dictionary
from helpers import reducer
from helpers import filehelper


class Configuration(object):
    output_directory = ''

    @staticmethod
    def initialize(configuration_file, working_directory=None, create_output=True, disable_merge=False):
        global configuration
        if not working_directory:
            working_directory = os.getcwd()

        if not configuration_file:
            raise EnvironmentError(
                "Configuration file " + configuration_file + "not found - aborting"
            )
        configuration_dict = Configuration.load_config(configuration_file)
        configuration = dictionary.to_named_tuple(configuration_dict)

        if create_output:
            Configuration.output_directory = os.path.join(working_directory,
                                                          Configuration.get(
                                                              'environment.output_path', optional=False),
                                                          datetime.datetime.now().strftime(
                                                              '%Y%m%d-%H%M%S') + "-" + os.path.basename(
                                                              configuration_file.replace(".jsonc", "")))
            if not os.path.exists(Configuration.output_directory):
                os.makedirs(Configuration.output_directory)
                copy(configuration_file, Configuration.output_directory)
        else:
            Configuration.output_directory = working_directory

    @staticmethod
    def load_config(configuration_path):
        if not os.path.exists(configuration_path):
            raise EnvironmentError(
                "Configuration file " + configuration_path + " not found - aborting.")
        with open(Configuration.build_path(configuration_path, create=False), 'r') as f:
            config = commentjson.load(f, object_hook=lambda d: {
                k: converter.try_eval(d[k]) for k in d})
            return config

    @staticmethod
    def get(key='', optional=True, default=None):
        global configuration
        if not configuration:
            raise EnvironmentError(
                "No configuration existing"
            )
        if key:
            try:
                value = reducer.rgetattr(configuration, key)
                if value is None:
                    if optional:
                        return default
                    raise EnvironmentError(
                        "Invalid key"
                    )
                return value
            except:
                if not optional:
                    raise EnvironmentError(
                        "Invalid key"
                    )
        return default

    @staticmethod
    def get_path(key='', create=True, optional=True):
        path = Configuration.get(key, optional)
        return Configuration.build_path(path)

    @staticmethod
    def build_path(path, create=True):
        """
        Generates an absolute path if a relative is passed.
        """
        path = filehelper.build_abspath(path, Configuration.output_directory)
        if create and not os.path.exists(path):
            os.makedirs(path)
        return path
