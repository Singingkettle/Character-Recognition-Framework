#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: encoder.py
@time: 2019/7/22 17:23
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import tensorflow as tf
from tensorflow.contrib import rnn

from blocks import basenet
from config.config import config
from stn.stn import get_stn
from encoder.shadownet import ShadowNet

# TODO: Add more cnn type for training and evaluation
cnn_factory = {
    'shadow': ShadowNet,
}


class Encoder(basenet.CNNBaseModel):
    """
        Implement the encoder model for feature extraction
    """

    def __init__(self, phase, hidden_nums, layers_nums, num_classes, cnn_type='shadow'):
        """

        :param phase: 'Train' or 'Test'
        """
        super(Encoder, self).__init__()
        if phase == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._hidden_nums = hidden_nums
        self._layers_nums = layers_nums
        self._num_classes = num_classes
        self._is_training = self._init_phase()
        self._cnn = self._get_cnn(cnn_type, phase)
        if config.ENCODER.SUMMARY_ACTIVATION and phase is 'train':
            self._summary = True
        else:
            self._summary = False
        # TODO: add more stn mode for training
        if config.USE_STN:
            self._stn = self._get_stn('tps', phase)
        else:
            self._stn = None

    def _init_phase(self):
        """
        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    @staticmethod
    def _get_stn(stn_type, phase):
        stn_net = get_stn(stn_type, phase)
        return stn_net

    @staticmethod
    def _get_cnn(cnn_type, phase):
        cnn = cnn_factory[cnn_type]

        return cnn(phase)

    def _cnn_extraction(self, inputdata, name):
        """ Implements section 2.1 of the paper: "Feature Sequence Extraction"

        :param inputdata: eg. batch*32*100*3 NHWC format
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            cnn_out = self._cnn.feature_sequence_extraction(inputdata)

        return cnn_out

    def _map_to_sequence(self, inputdata, name):
        """ Implements the map to sequence part of the network.

        This is used to convert the CNN feature map to the sequence used in the stacked LSTM layers later on.
        Note that this determines the length of the sequences that the LSTM expects
        :param inputdata:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            shape = inputdata.get_shape().as_list()
            assert shape[1] == 1  # H of the feature map must equal to 1

            ret = self.squeeze(inputdata=inputdata, axis=1, name='squeeze')

        return ret

    def _sequence_lstm(self, inputdata, name):
        """ Implements the sequence label part of the network

        :param inputdata:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]
            # Backward direction cells
            bw_cell_list = [tf.nn.rnn_cell.LSTMCell(nh, forget_bias=1.0) for
                            nh in [self._hidden_nums] * self._layers_nums]

            stack_lstm, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list, bw_cell_list, inputdata,
                dtype=tf.float32
            )
            stack_lstm = self.dropout(
                inputdata=stack_lstm,
                keep_prob=0.5,
                is_training=self._is_training,
                name='sequence_drop_out'
            )

        return stack_lstm

    def forward(self, input_data, name):
        """
        Main routine to construct the network
        :param input_data:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # centerlized data
            if self._summary:
                tf.summary.image('Encoder_Inputs_With_Normal', input_data[:1], max_outputs=1)
            if config.USE_STN:
                input_data = self._stn.forward(input_data)
            input_data = tf.subtract(tf.divide(input_data, 127.5), 1.0)

            # first apply the cnn feature extraction stage
            cnn_out = self._cnn_extraction(
                inputdata=input_data, name='feature_extraction_module'
            )

            # second apply the map to sequence stage
            sequence = self._map_to_sequence(
                inputdata=cnn_out, name='map_to_sequence_module'
            )

            # third apply the sequence lstm stage
            net_out = self._sequence_lstm(
                inputdata=sequence, name='sequence_rnn_module'
            )

        return net_out


def get_encoder(phase):
    encoder_net = Encoder(phase=phase, hidden_nums=config.ENCODER.HIDDEN_UNITS,
                          layers_nums=config.ENCODER.HIDDEN_LAYERS, num_classes=config.DATASET.NUM_CLASSES,
                          cnn_type=config.ENCODER.NETWORKTYPE)

    return encoder_net
