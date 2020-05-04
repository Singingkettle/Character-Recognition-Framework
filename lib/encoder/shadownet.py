#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: shadownet.py
@time: 2019/7/22 17:23
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import tensorflow as tf

from blocks import basenet
from config.config import config


class ShadowNet(basenet.CNNBaseModel):
    """
        Implement the crnn model for squence recognition
    """

    def __init__(self, phase):
        """

        :param phase: 'Train' or 'Test'
        """
        super(ShadowNet, self).__init__()
        if phase == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    def _conv_stage(self, inputdata, out_dims, name):
        """ Standard VGG convolutional stage: 2d conv, relu, and maxpool

        :param inputdata: 4D tensor batch x width x height x channels
        :param out_dims: number of output channels / filters
        :return: the maxpooled output of the stage
        """
        with tf.variable_scope(name_or_scope=name):
            conv = self.conv2d(
                inputdata=inputdata, out_channel=out_dims,
                kernel_size=3, stride=1, use_bias=True, name='conv'
            )
            bn = self.layerbn(
                inputdata=conv, is_training=self._is_training, name='bn'
            )
            relu = self.relu(
                inputdata=bn, name='relu'
            )
            max_pool = self.maxpooling(
                inputdata=relu, kernel_size=2, stride=2, name='max_pool'
            )
        return max_pool

    def feature_sequence_extraction(self, inputdata):
        """ Implements section 2.1 of the paper: "Feature Sequence Extraction"

        :param inputdata: eg. batch*32*100*3 NHWC format
        :return:
        """
        with tf.variable_scope(name_or_scope='shadow_net'):
            conv1 = self._conv_stage(
                inputdata=inputdata, out_dims=64, name='conv1'
            )
            conv2 = self._conv_stage(
                inputdata=conv1, out_dims=128, name='conv2'
            )
            conv3 = self.conv2d(
                inputdata=conv2, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv3'
            )
            bn3 = self.layerbn(
                inputdata=conv3, is_training=self._is_training, name='bn3'
            )
            relu3 = self.relu(
                inputdata=bn3, name='relu3'
            )
            conv4 = self.conv2d(
                inputdata=relu3, out_channel=256, kernel_size=3, stride=1, use_bias=False, name='conv4'
            )
            bn4 = self.layerbn(
                inputdata=conv4, is_training=self._is_training, name='bn4'
            )
            relu4 = self.relu(
                inputdata=bn4, name='relu4')
            max_pool4 = self.maxpooling(
                inputdata=relu4, kernel_size=[2, 1], stride=[2, 1], padding='VALID', name='max_pool4'
            )
            conv5 = self.conv2d(
                inputdata=max_pool4, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv5'
            )
            bn5 = self.layerbn(
                inputdata=conv5, is_training=self._is_training, name='bn5'
            )
            relu5 = self.relu(
                inputdata=bn5, name='bn5'
            )
            conv6 = self.conv2d(
                inputdata=relu5, out_channel=512, kernel_size=3, stride=1, use_bias=False, name='conv6'
            )
            bn6 = self.layerbn(
                inputdata=conv6, is_training=self._is_training, name='bn6'
            )
            relu6 = self.relu(
                inputdata=bn6, name='relu6'
            )
            max_pool6 = self.maxpooling(
                inputdata=relu6, kernel_size=[2, 1], stride=[2, 1], name='max_pool6'
            )
            conv7 = self.conv2d(
                inputdata=max_pool6, out_channel=512, kernel_size=2, stride=[2, 1], use_bias=False, name='conv7'
            )
            bn7 = self.layerbn(
                inputdata=conv7, is_training=self._is_training, name='bn7'
            )
            relu7 = self.relu(
                inputdata=bn7, name='bn7'
            )

            # TODO: ADD the tensor, which you want to keep an eye
            # if config.ENCODER.SUMMARY_ACTIVATION:
            #     tf.summary.histogram('ShadowNet/Relu7', relu7)

        return relu7
