#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/6/5 21:04
@Author  : Citybuster
@Site    : https://www.zhihu.com/people/chang-shuo-59/activities
@File    : net.py
@Software: PyCharm
@Email: changshuo@bupt.edu.cn
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from config.config import config
from blocks import basenet


class TPSNet(basenet.CNNBaseModel):
    def __init__(self, batch_size, num_control_points=None, init_bias_pattern=None,
                 margins=None, activation=None, is_summary=None):
        """
        :param batch_size:
        :param num_control_points:
        :param init_bias_pattern:
        :param margins:
        :param activation:
        :param is_summary:
        """
        super(TPSNet, self).__init__()
        self._batch_size = batch_size
        self._num_control_points = num_control_points
        self._is_summary = is_summary
        self._init_bias = self._build_init_bias(init_bias_pattern, margins, activation)

    def _build_init_bias(self, pattern, margins, activation):
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = self._num_control_points // 2
        upper_x = np.linspace(margin_x, 1.0 - margin_x, num=num_ctrl_pts_per_side)
        lower_x = np.linspace(margin_x, 1.0 - margin_x, num=num_ctrl_pts_per_side)

        if pattern == 'slope':
            upper_y = np.linspace(margin_y, 0.3, num=num_ctrl_pts_per_side)
            lower_y = np.linspace(0.7, 1.0 - margin_y, num=num_ctrl_pts_per_side)
        elif pattern == 'identity':
            upper_y = np.linspace(margin_y, margin_y, num=num_ctrl_pts_per_side)
            lower_y = np.linspace(1.0 - margin_y, 1.0 - margin_y, num=num_ctrl_pts_per_side)
        elif pattern == 'sine':
            upper_y = 0.25 + 0.2 * np.sin(2 * np.pi * upper_x)
            lower_y = 0.75 + 0.2 * np.sin(2 * np.pi * lower_x)
        else:
            raise ValueError('Unknown initialization pattern: {}'.format(pattern))

        init_ctrl_pts = np.concatenate([
            np.stack([upper_x, upper_y], axis=1),
            np.stack([lower_x, lower_y], axis=1),
        ], axis=0)

        if activation == 'sigmoid':
            init_biases = -np.log(1. / init_ctrl_pts - 1.)
        elif activation == 'none':
            init_biases = init_ctrl_pts
        else:
            raise ValueError('Unknown activation type: {}'.format(activation))

        return init_biases

    def _conv_stage(self, input_data, out_dims, name):
        """ Standard VGG convolutional stage: 2d conv, relu, maxpool

        :param input_data: 4D tensor batch x width x height x channels
        :param out_dims: number of output channels / filters
        :return: the maxpooled output of the stage
        """
        with tf.variable_scope(name_or_scope=name):
            conv = self.conv2d(
                inputdata=input_data, out_channel=out_dims,
                kernel_size=3, stride=1, use_bias=True, name='conv'
            )
            relu = self.relu(
                inputdata=conv, name='relu'
            )
            max_pool = self.maxpooling(
                inputdata=relu, kernel_size=2, stride=2, name='max_pool'
            )
        return max_pool

    def forward(self, input_data):
        """Extract features
        :param input_data: 4D tensor batch x width x height x channels
        :param reuse:
        :return: the control points
        """
        with tf.variable_scope(name_or_scope='LocalizationNetwork'):
            conv1 = self._conv_stage(
                input_data=input_data, out_dims=32, name='conv1'
            )
            conv2 = self._conv_stage(
                input_data=conv1, out_dims=64, name='conv2'
            )
            conv3 = self._conv_stage(
                input_data=conv2, out_dims=128, name='conv3'
            )
            conv4 = self._conv_stage(
                input_data=conv3, out_dims=256, name='conv4'
            )
            conv5 = self._conv_stage(
                input_data=conv4, out_dims=256, name='conv5'
            )
            conv6 = self.conv2d(
                inputdata=conv5, out_channel=256, kernel_size=3, stride=[1, 1], use_bias=True, name='conv6'
            )
            conv_output = tf.reshape(conv6, [self._batch_size, -1])
            fc1 = fully_connected(conv_output, 512)
            fc2_weights_initializer = tf.zeros_initializer()
            fc2_biases_initializer = tf.constant_initializer(self._init_bias)
            fc2 = fully_connected(0.1 * fc1, 2 * self._num_control_points,
                                  weights_initializer=fc2_weights_initializer,
                                  biases_initializer=fc2_biases_initializer,
                                  activation_fn=None,
                                  normalizer_fn=None)
            ctrl_pts = tf.sigmoid(fc2)

            ctrl_pts = tf.reshape(ctrl_pts, [self._batch_size, self._num_control_points, 2])

            if self._is_summary:
                tf.summary.histogram('TPS_STN_FC1', fc1)
                tf.summary.histogram('TPS_STN_FC2', fc2)

        return ctrl_pts
