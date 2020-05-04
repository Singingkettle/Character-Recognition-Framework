#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/6/5 22:11
@Author  : Citybuster
@Site    : https://www.zhihu.com/people/chang-shuo-59/activities
@File    : grid.py
@Software: PyCharm
@Email: changshuo@bupt.edu.cn
"""

import numpy as np
import tensorflow as tf

from blocks import basenet

eps = 1e-6


class TPSGrid(basenet.CNNBaseModel):
    def __init__(self, batch_size, num_control_points=None, output_image_size=None, margins=None):
        """
        :param batch_size:
        :param num_control_points:
        :param output_image_size:
        :param margins:
        """
        super(TPSGrid, self).__init__()
        self._batch_size = batch_size
        self._num_control_points = num_control_points
        self._output_image_size = output_image_size

        self._output_grid = self._build_output_grid()
        self._output_ctrl_pts = self._build_output_control_points(margins)
        self._inv_delta_c = self._build_helper_constants()

    def _build_output_control_points(self, margins):
        margin_x, margin_y = margins
        num_ctrl_pts_per_side = self._num_control_points // 2
        ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
        ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
        ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        output_ctrl_pts = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return output_ctrl_pts

    def _build_helper_constants(self):
        C = self._output_ctrl_pts
        k = self._num_control_points
        hat_C = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                hat_C[i, j] = np.linalg.norm(C[i] - C[j])
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        delta_C = np.concatenate(
            [
                np.concatenate([np.ones((k, 1)), C, hat_C], axis=1),
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),
                np.concatenate([np.zeros((1, 3)), np.ones((1, k))], axis=1)
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C

    def _build_output_grid(self):
        output_h, output_w = self._output_image_size
        output_grid_x = (np.arange(output_w) + 0.5) / output_w
        output_grid_y = (np.arange(output_h) + 0.5) / output_h
        output_grid = np.stack(
            np.meshgrid(output_grid_x, output_grid_y),
            axis=2)
        return output_grid  # Ｈ * W * 2

    def forward(self, input_data, reuse=False):
        """Extract features
        :param input_data: 4D tensor batch x width x height x channels
        :param reuse:
        :return: the control points
        """
        with tf.variable_scope(name_or_scope='GenerateGrid', reuse=reuse):
            C = tf.constant(self._output_ctrl_pts, tf.float32)  # => [k, 2]
            batch_Cp = input_data  # => [B, k, 2]

            inv_delta_c = tf.constant(self._inv_delta_c, dtype=tf.float32)
            batch_inv_delta_c = tf.tile(
                tf.expand_dims(inv_delta_c, 0),
                [self._batch_size, 1, 1])  # => [B, k+3, k+3]
            batch_Cp_zero = tf.concat(
                [batch_Cp, tf.zeros([self._batch_size, 3, 2])],
                axis=1)  # => [B, k+3, 2]
            batch_T = tf.matmul(batch_inv_delta_c, batch_Cp_zero)  # => [B, k+3, 2]

            k = self._num_control_points
            G = tf.constant(self._output_grid.reshape([-1, 2]), tf.float32)  # => [n, 2]
            n = G.shape[0]

            G_tile = tf.tile(tf.expand_dims(G, axis=1), [1, k, 1])  # => [n,k,2]
            C_tile = tf.expand_dims(C, axis=0)  # => [1, k, 2]
            G_diff = G_tile - C_tile  # => [n, k, 2]
            rbf_norm = tf.norm(G_diff, axis=2, ord=2, keepdims=False)  # => [n, k]
            rbf = tf.multiply(tf.square(rbf_norm), tf.log(rbf_norm + eps))  # => [n, k]
            G_lifted = tf.concat([tf.ones([n, 1]), G, rbf], axis=1)  # => [n, k+3]
            batch_G_lifted = tf.tile(tf.expand_dims(G_lifted, 0), [self._batch_size, 1, 1])  # => [B, n, k+3]

            batch_Gp = tf.matmul(batch_G_lifted, batch_T)

        return batch_Gp
