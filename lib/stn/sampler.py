#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/6/5 22:22
@Author  : Citybuster
@Site    : https://www.zhihu.com/people/chang-shuo-59/activities
@File    : sampler.py
@Software: PyCharm
@Email: changshuo@bupt.edu.cn
"""

import tensorflow as tf

from blocks import basenet
from config.config import config
from local_utils import utils

eps = 1e-6


class TPSSampler(basenet.CNNBaseModel):
    def __init__(self, batch_size, num_control_points=None, input_image_size=None,
                 output_image_size=None, is_summary=None):
        """
        :param batch_size:
        :param num_control_points:
        :param input_image_size:
        :param output_image_size:
        :param is_summary:
        """
        super(TPSSampler, self).__init__()
        self._batch_size = batch_size
        self._num_control_points = num_control_points
        self._input_image_size = input_image_size
        self._output_image_size = output_image_size
        self._is_summary = is_summary

    @staticmethod
    def _get_pixels(images, batch_x, batch_y, batch_indices):
        indices = tf.stack([batch_indices, batch_y, batch_x], axis=2)  # => [B, n, 3]
        pixels = tf.gather_nd(images, indices)
        return pixels

    def forward(self, input_data, sampling_grid):
        """Extract features
        :param input_data: 4D tensor batch x width x height x channels
        :param sampling_grid:
        :return: the control points
        """
        with tf.variable_scope(name_or_scope='Sampler'):
            if input_data.dtype != tf.float32:
                raise ValueError('image must be of type tf.float32')
            batch_G = sampling_grid
            batch_size = self._batch_size
            image_h, image_w = self._input_image_size
            n = utils.combined_static_and_dynamic_shape(sampling_grid)[1]

            batch_Gx = image_w * batch_G[:, :, 0]
            batch_Gy = image_h * batch_G[:, :, 1]
            batch_Gx = tf.clip_by_value(batch_Gx, 0., image_w - 2)
            batch_Gy = tf.clip_by_value(batch_Gy, 0., image_h - 2)

            batch_Gx0 = tf.cast(tf.floor(batch_Gx), tf.int32)  # G* => [batch_size, n, 2]
            batch_Gx1 = batch_Gx0 + 1  # G*x, G*y => [batch_size, n]
            batch_Gy0 = tf.cast(tf.floor(batch_Gy), tf.int32)
            batch_Gy1 = batch_Gy0 + 1

            batch_indices = tf.tile(
                tf.expand_dims(tf.range(batch_size), 1),
                [1, n])  # => [B, n]
            batch_I00 = self._get_pixels(input_data, batch_Gx0, batch_Gy0, batch_indices)
            batch_I01 = self._get_pixels(input_data, batch_Gx0, batch_Gy1, batch_indices)
            batch_I10 = self._get_pixels(input_data, batch_Gx1, batch_Gy0, batch_indices)
            batch_I11 = self._get_pixels(input_data, batch_Gx1, batch_Gy1, batch_indices)  # => [B, n, d]

            batch_Gx0 = tf.to_float(batch_Gx0)
            batch_Gx1 = tf.to_float(batch_Gx1)
            batch_Gy0 = tf.to_float(batch_Gy0)
            batch_Gy1 = tf.to_float(batch_Gy1)

            batch_w00 = (batch_Gx1 - batch_Gx) * (batch_Gy1 - batch_Gy)
            batch_w01 = (batch_Gx1 - batch_Gx) * (batch_Gy - batch_Gy0)
            batch_w10 = (batch_Gx - batch_Gx0) * (batch_Gy1 - batch_Gy)
            batch_w11 = (batch_Gx - batch_Gx0) * (batch_Gy - batch_Gy0)  # => [B, n]

            batch_pixels = tf.add_n([
                tf.expand_dims(batch_w00, axis=2) * batch_I00,
                tf.expand_dims(batch_w01, axis=2) * batch_I01,
                tf.expand_dims(batch_w10, axis=2) * batch_I10,
                tf.expand_dims(batch_w11, axis=2) * batch_I11,
            ])

            output_h, output_w = self._output_image_size
            output_maps = tf.reshape(batch_pixels, [batch_size, output_h, output_w, -1])
            output_maps = tf.cast(output_maps, dtype=input_data.dtype)

            if self._is_summary:
                tf.summary.image('TPS_InputImage', input_data[:1], max_outputs=1)
                tf.summary.image('TPS_RectifiedImage', output_maps[:1], max_outputs=1)

        return output_maps
