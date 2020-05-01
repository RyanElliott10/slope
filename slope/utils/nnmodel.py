# -*- coding: utf-8 -*-
# Base class for all data points that will be fed through a neural network. Acts moreso as an
# interface than a superclass.
#
# Author: Ryan Elliott <ryan.elliott31@gmail.com>
#
# For license information, see LICENSE

class NNModel(object):
    """
    Mostly an interface for all neural network models to conform to
    """

    def __init__(self, data):
        self.data = data

    @property
    def nn_format(self):
        raise NotImplementedError

    @staticmethod
    def make_id(major: int, minor: int) -> str:
        return f'{major}.{minor}'
