#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: decoder_ctc.py
@time: 2019/7/23 14:12
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import tensorflow as tf

from config.config import config
from decoder import decoder_base


class CTCDecoder(decoder_base.DecoderBase):
    """
        Implement the ctc decoder for sequence recognition
    """

    def __init__(self, phase, sequence_length, decode_mode):
        """
        :param phase: 'Train' or 'Test'
        :param sequence_length:
        :param decode_mode:
        """
        super(CTCDecoder, self).__init__()
        if phase == 'test':
            self._is_training = False
        else:
            self._is_training = True
        self._sequence_length = sequence_length
        self._merge_repeated = False
        self._predictor = 'CTC_Decoder'
        self._decode_mode = decode_mode
        self.phase = phase

    def predict(self, input_data):
        with tf.variable_scope(name_or_scope=self._decode_mode + '_' + self._predictor):
            shape = tf.shape(input_data)
            rnn_reshaped = tf.reshape(input_data, [shape[0] * shape[1], shape[2]])
            w = tf.get_variable(
                name='w',
                shape=[config.ENCODER.HIDDEN_UNITS * 2, config.DATASET.NUM_CLASSES],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                trainable=True
            )

            # Doing the affine projection
            logits = tf.matmul(rnn_reshaped, w, name='logits')
            logits = tf.reshape(logits, [shape[0], shape[1], config.DATASET.NUM_CLASSES], name='logits_reshape')

            # Swap batch and batch axis
            logits = tf.transpose(logits, [1, 0, 2], name='transpose_time_major')  # [width, batch, n_classes]

            # Perform ctc beam search
            prediction_labels, prediction_scores = tf.nn.ctc_beam_search_decoder(
                logits,
                self._sequence_length,
                merge_repeated=self._merge_repeated
            )

            outputs_dict = {
                'logits': logits,
                'labels': prediction_labels[0],
                'scores': prediction_scores
            }

        return outputs_dict

    def loss(self, input_data):
        """
        :param input_data:
        :return: loss
        """
        with tf.name_scope(self.phase + '_' + self._decode_mode + '_Loss'):
            loss = tf.reduce_mean(
                tf.nn.ctc_loss(
                    labels=self._label, inputs=input_data['logits'],
                    sequence_length=self._sequence_length
                ),
                name=self.phase + '_' + self._predictor + '_ctc_loss'
            )

        return loss

    def sequence_dist(self, input_data):
        """
        :param input_data:
        :return: seq_dis
        """
        with tf.name_scope(self.phase + '_' + self._decode_mode + '_Sequence_Dist'):
            seq_dis = tf.reduce_mean(
                tf.edit_distance(tf.cast(input_data['labels'], tf.int32), self._label),
                name='train_edit_distance'
            )

        return seq_dis


class NormalCTC(object):
    """
        Implement the ctc decoder for sequence recognition, which only has one decoding mode (from left to right).
        And we implement this extra class to keep the same framework with seqtoseq decoder (STS-Decoder).
    """
    def __init__(self, phase, sequence_length):
        """
        :param phase: 'Train' or 'Test'
        :param sequence_length:
        """
        self._decoder = CTCDecoder(phase, sequence_length, 'Normal')

    def set_label(self, label):
        self._decoder.set_label(label['LeftToRight'])

    def predict(self, input_data):

        outputs_dict = self._decoder.predict(input_data)

        return outputs_dict

    def loss(self, input_data):

        loss = self._decoder.loss(input_data)
        return loss

    def sequence_dist(self, input_data):

        seq_dis = self._decoder.sequence_dist(input_data)

        return seq_dis

