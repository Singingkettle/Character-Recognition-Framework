#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: decoder_base.py
@time: 2019/7/22 17:21
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

from abc import ABCMeta
from abc import abstractmethod


class DecoderBase(object):
    """
    Base model for decoder
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._predictor = 'decoder'
        self._label = None
        pass

    @abstractmethod
    def set_label(self, label):
        self._label = label

    @abstractmethod
    def predict(self, input_data):
        pass

    @abstractmethod
    def loss(self, input_data):
        pass

    @abstractmethod
    def sequence_dist(self, input_data):
        pass
