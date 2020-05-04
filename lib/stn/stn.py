#!/usr/bin/env python
# -*- coding: utf-8 -*
"""
@Time    : 2019/6/5 20:33
@Author  : Citybuster
@Site    : https://www.zhihu.com/people/chang-shuo-59/activities
@File    : stn.py
@Software: PyCharm
@Email: changshuo@bupt.edu.cn
"""

import tensorflow as tf

from config.config import config
from blocks import basenet
from stn.grid import TPSGrid
from stn.net import TPSNet
from stn.sampler import TPSSampler

# TODO: Add more modes for stn
localization_factory = {
    'tps': TPSNet,
}

grid_factory = {
    'tps': TPSGrid,
}

sampler_factory = {
    'tps': TPSSampler,
}


class STNNet(basenet.CNNBaseModel):
    def __init__(self, phase, batch_size, num_control_points=None, input_image_size=None, output_image_size=None,
                 localization_image_size=None, init_bias_pattern=None, margins=None, activation='none', stn_type='tps'):
        """
        :param phase: 'Train' or 'Test'
        :param batch_size:
        :param num_control_points:
        :param input_image_size:
        :param output_image_size:
        :param init_bias_pattern:
        :param margins
        :param activation
        :param stn_type
        """
        super(STNNet, self).__init__()
        if phase == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._is_training = self._init_phase()
        self._localization_image_size = localization_image_size
        if config.STN.SUMMARY_ACTIVATION and phase is 'train':
            self._summary = True
        else:
            self._summary = False
        self._localization = self._get_localization(stn_type, batch_size, num_control_points, init_bias_pattern,
                                                    margins, activation, self._summary)
        self._grid = self._get_grid(stn_type, batch_size, num_control_points, output_image_size, margins)
        self._sampler = self._get_sampler(stn_type, batch_size, num_control_points,
                                          input_image_size, output_image_size, self._summary)

    def _init_phase(self):
        """
        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    @staticmethod
    def _get_localization(cnn_type, batch_size, num_control_points,
                          init_bias_pattern, margins, activation, is_summary):
        localization = localization_factory[cnn_type]

        return localization(batch_size, num_control_points, init_bias_pattern, margins, activation, is_summary)

    @staticmethod
    def _get_grid(grid_type, batch_size, num_control_points, output_image_size, margins):
        grid = grid_factory[grid_type]

        return grid(batch_size, num_control_points, output_image_size, margins)

    @staticmethod
    def _get_sampler(sampler_type, batch_size, num_control_points, input_image_size, output_image_size, is_summary):
        sampler = sampler_factory[sampler_type]

        return sampler(batch_size, num_control_points, input_image_size, output_image_size, is_summary)

    def forward(self, input_data):
        """Extract features
        :param input_data: 4D tensor batch x width x height x channels
        :return: the control points
        """
        with tf.variable_scope(name_or_scope='STN'):
            resize_images = tf.image.resize_images(input_data, self._localization_image_size)
            input_control_points = self._localization.forward(resize_images)
            sampling_grid = self._grid.forward(input_control_points)
            rectified_images = self._sampler.forward(input_data, sampling_grid)

        return rectified_images


def get_stn(stn_type, phase):
    if phase == 'test':
        batch_size = 1
    else:
        batch_size = config.TRAIN.BATCH_SIZE
    stn_net = STNNet(phase, config.TRAIN.BATCH_SIZE, config.STN.NCP, (config.STN.IH, config.STN.IW),
                     (config.STN.OH, config.STN.OW), (config.STN.ILH, config.STN.ILW), config.STN.IB,
                     (config.STN.MY, config.STN.MX), activation='none', stn_type=stn_type)
    return stn_net