#!/usr/bin/env python3
# coding: utf8

"""
Converts and handles data and types.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import datetime
import time


def try_eval(expression):
    """
    Returns an evaluated expression if possible.
    If not evaluatable the expression is returned.
    """
    if expression:
        try:
            return eval(expression)
        except:
            pass
    return expression


def str2bool(value):
    """
    Converts a string to a boolean value.
    """
    if type(value) == bool:
        return value
    return value and value.lower() in ('yes', 'true', 't', '1', 'y')


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def elapsed_since(start, format="%H:%M:%S"):
    return time.strftime(format, time.gmtime(time.time() - start))


def to_complex(realimag):
    'Converts the real-imag array of the training values to complex values'
    real = realimag[:, :, 0]
    imag = realimag[:, :, 1]
    return real + imag * 1j
