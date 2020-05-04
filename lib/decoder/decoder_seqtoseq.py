#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: decoder_seqtoseq.py
@time: 2019/7/22 17:22
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import functools

import tensorflow as tf
from tensorflow.contrib import seq2seq

from config.config import config
from decoder import decoder_base
from local_utils.utils import combined_static_and_dynamic_shape
from wrapper import sync_attention_wrapper


def provide_groundtruth(label, batch_size, start_label, end_label):
    groundtruth_dict = dict()
    with tf.name_scope('Convert_label', 'ProvideGroundtruth', [label]):
        # convert label
        dense_label = tf.sparse.to_dense(label, default_value=end_label)

        # get the txt length
        text_lengths = tf.sparse_reduce_sum(
            tf.SparseTensor(
                label.indices,
                tf.fill([tf.shape(label.indices)[0]], 1),
                label.dense_shape
            ),
            axis=1
        )
        text_lengths.set_shape([None])

        start_label = tf.fill([batch_size, 1], tf.constant(start_label, tf.int32))
        end_label = tf.fill([batch_size, 1], tf.constant(end_label, tf.int32))
        decoder_inputs = tf.concat([start_label, dense_label], axis=1)
        decoder_targets = tf.concat([dense_label, end_label], axis=1)
        decoder_lengths = text_lengths + 1

        # set maximum lengths
        decoder_inputs = decoder_inputs[:, :config.DECODER_STS.MNS]
        decoder_targets = decoder_targets[:, :config.DECODER_STS.MNS]
        decoder_lengths = tf.minimum(decoder_lengths, config.DECODER_STS.MNS)

        groundtruth_dict['decoder_inputs'] = decoder_inputs
        groundtruth_dict['decoder_targets'] = decoder_targets
        groundtruth_dict['decoder_lengths'] = decoder_lengths

        return groundtruth_dict


class STSDecoder(decoder_base.DecoderBase):
    """
        Implement the ctc decoder for sequence recognition
    """

    def __init__(self, phase, sequence_length, decode_mode):
        """
        :param phase: 'Train' or 'Test'
        :param sequence_length:
        """
        super(STSDecoder, self).__init__()
        if phase == 'train':
            self._is_training = True
        else:
            self._is_training = False
        if phase == 'test':
            self._batch_size = 1
        else:
            self._batch_size = config.TRAIN.BATCH_SIZE

        self._sequence_length = sequence_length
        self._predictor = 'STS_Decoder'
        self._sequence_normalize = False
        self._sample_normalize = True
        self._initializer_mode = config.DECODER_STS.I
        self._weight = 0.5
        self._phase = phase
        self._decode_mode = decode_mode
        self._rnn_regularizer = self._build_regularizer(config.DECODER_STS.R)
        self._rnn_cell = self._build_rnn_cell()
        self._groundtruth_dict = None

    @property
    def start_label(self):
        return 36  # 表示开始符

    @property
    def end_label(self):
        return 37  # 表示停止符

    def set_groundtruth_dict(self):
        self._groundtruth_dict = provide_groundtruth(self._label, self._batch_size, self.start_label, self.end_label)

    @staticmethod
    def _build_regularizer(regularizer):
        if regularizer == 'l1_regularizer':
            return tf.contrib.layers.l1_regularizer(scale=float(config.DECODER_STS.L1W))
        if regularizer == 'l2_regularizer':
            return tf.contrib.layers.l2_regularizer(scale=float(config.DECODER_STS.L2W))
        raise ValueError('Unknown regularizer function: {}'.format(regularizer))

    @staticmethod
    def _build_initializer(initializer):
        if initializer == 'truncated_normal_initializer':
            return tf.truncated_normal_initializer(
                mean=config.DECODER_STS.TM,
                stddev=config.DECODER_STS.TS)
        if initializer == 'orthogonal_initializer':
            return tf.orthogonal_initializer(
                gain=config.DECODER_STS.OG,
                seed=config.DECODER_STS.OS
            )
        if initializer == 'uniform_initializer':
            return tf.random_uniform_initializer(
                minval=config.DECODER_STS.UMI,
                maxval=config.DECODER_STS.UMA)
        raise ValueError('Unknown initializer function: {}'.format(
            initializer))

    def _build_rnn_cell(self):
        # TODO: other rnn cell
        weights_initializer_object = self._build_initializer(config.DECODER_STS.I)
        rnn_cell_object = tf.contrib.rnn.LSTMCell(
            config.DECODER_STS.NU,
            use_peepholes=False,
            forget_bias=config.DECODER_STS.FB,
            initializer=weights_initializer_object
        )
        return rnn_cell_object

    def _build_attention_mechanism(self, input_data):
        """Build (possibly multiple) attention mechanisms."""
        if self._phase == 'test':
            input_data = seq2seq.tile_batch(input_data, multiplier=config.DECODER_STS.BW)

        return seq2seq.BahdanauAttention(config.DECODER_STS.NU, input_data, memory_sequence_length=None)

    def _build_decoder_cell(self, input_data):
        attention_mechanism = self._build_attention_mechanism(input_data)
        wrapper_class = sync_attention_wrapper.SyncAttentionWrapper
        decoder_cell = wrapper_class(
            self._rnn_cell,
            attention_mechanism,
            output_attention=False)

        return decoder_cell

    def _build_decoder(self, decoder_cell, batch_size):
        embedding_fn = functools.partial(tf.one_hot, depth=config.DATASET.NUM_CLASSES)
        output_layer = tf.layers.Dense(
            config.DATASET.NUM_CLASSES,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer())
        if self._phase != 'test':
            train_helper = seq2seq.TrainingHelper(
                embedding_fn(self._groundtruth_dict['decoder_inputs']),
                sequence_length=self._groundtruth_dict['decoder_lengths'],
                time_major=False)  # Teacher Forcing
            decoder = seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=train_helper,
                initial_state=decoder_cell.zero_state(batch_size, tf.float32),
                output_layer=output_layer)
        else:
            decoder = seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=embedding_fn,
                start_tokens=tf.fill([self._batch_size], self.start_label),
                end_token=self.end_label,
                initial_state=decoder_cell.zero_state(self._batch_size * config.DECODER_STS.BW, tf.float32),
                beam_width=config.DECODER_STS.BW,
                output_layer=output_layer,
                length_penalty_weight=0.0)

        return decoder

    def predict(self, input_data):

        # Build decoder cell
        with tf.variable_scope(name_or_scope=self._decode_mode + '_' + self._predictor):
            decoder_cell = self._build_decoder_cell(input_data)
            decoder = self._build_decoder(decoder_cell, self._batch_size)

            outputs, _, output_lengths = seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=config.DECODER_STS.MNS
            )
            # Apply regularizer
            filter_weights = lambda vars: [x for x in vars if x.op.name.endswith('kernel')]
            tf.contrib.layers.apply_regularization(
                self._rnn_regularizer,
                filter_weights(decoder_cell.trainable_weights))

            if self._phase != 'test':
                assert isinstance(outputs, seq2seq.BasicDecoderOutput)
                indices = tf.where(tf.less(outputs.sample_id, self.start_label))
                values = tf.gather_nd(outputs.sample_id, indices)
                shape = tf.shape(outputs.sample_id, out_type=tf.int64)
                sparse_labels = tf.SparseTensor(indices, values, dense_shape=shape)
                outputs_dict = {
                    'labels': sparse_labels,
                    'logits': outputs.rnn_output,
                }
            else:
                assert isinstance(outputs, seq2seq.FinalBeamSearchDecoderOutput)
                prediction_labels = outputs.predicted_ids[:, :, 0]
                prediction_lengths = output_lengths[:, 0]
                prediction_scores = tf.gather_nd(
                    outputs.beam_search_decoder_output.scores[:, :, 0],
                    tf.stack([tf.range(self._batch_size), prediction_lengths - 1], axis=1)
                )
                indices = tf.where(tf.less(prediction_labels, self.start_label))
                values = tf.gather_nd(prediction_labels, indices)
                shape = tf.shape(prediction_labels, out_type=tf.int64)
                sparse_labels = tf.SparseTensor(indices, values, dense_shape=shape)
                outputs_dict = {
                    'labels': sparse_labels,
                    'scores': prediction_scores,
                    'lengths': prediction_lengths
                }

        return outputs_dict

    def loss(self, input_data):
        """
        :param input_data:
        :return: loss
        """
        with tf.name_scope(self._phase + '_' + self._decode_mode + '_Loss'):
            raw_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._groundtruth_dict['decoder_targets'],
                logits=input_data['logits']
            )
            _, max_time = combined_static_and_dynamic_shape(self._groundtruth_dict['decoder_targets'])
            mask = tf.less(
                tf.tile([tf.range(max_time)], [self._batch_size, 1]),
                tf.expand_dims(self._groundtruth_dict['decoder_lengths'], 1),
                name='mask'
            )
            masked_losses = tf.multiply(
                raw_losses,
                tf.cast(mask, tf.float32),
                name='masked_losses'
            )  # => [batch_size, max_time]
            row_losses = tf.reduce_sum(masked_losses, 1, name='row_losses')
            loss = tf.reduce_sum(row_losses)
            if self._sample_normalize:
                loss = tf.truediv(loss, tf.cast(tf.maximum(self._groundtruth_dict['decoder_lengths'], 1), tf.float32))
            if self._weight:
                loss = loss * self._weight

            loss = tf.reduce_mean(loss, name=self._phase + '_' + self._predictor + '_sts_loss')
            tf.losses.add_loss(loss)

        return loss

    def sequence_dist(self, input_data):
        """
        :param input_data:
        :return: seq_dis
        """
        with tf.name_scope(self._phase + '_' + self._decode_mode + '_Sequence_Dist'):
            seq_dis = tf.reduce_mean(
                tf.edit_distance(tf.cast(input_data['labels'], tf.int32), self._label),
                name='train_edit_distance'
            )

        return seq_dis


class NormalSTS(object):
    """
        Implement the sts decoder for sequence recognition, which only has one decoding mode (from left to right).
    """

    def __init__(self, phase, sequence_length):
        """
        :param phase: 'Train' or 'Test'
        :param sequence_length:
        """
        self._decoder = STSDecoder(phase, sequence_length, 'Normal')

    def set_label(self, label):
        self._decoder.set_label(label['LeftToRight'])
        self._decoder.set_groundtruth_dict()

    def predict(self, input_data):
        outputs_dict = self._decoder.predict(input_data)

        return outputs_dict

    def loss(self, input_data):
        loss = self._decoder.loss(input_data)

        return loss

    def sequence_dist(self, input_data):
        seq_dis = self._decoder.sequence_dist(input_data)

        return seq_dis


class ReverseSTS(object):
    """
        Implement the sts decoder for sequence recognition, which only has one decoding mode (from left to right).
    """

    def __init__(self, phase, sequence_length):
        """
        :param phase: 'Train' or 'Test'
        :param sequence_length:
        """
        self._decoder = STSDecoder(phase, sequence_length, 'Reverse')

    def set_label(self, label):
        self._decoder.set_label(label['RightToLeft'])
        self._decoder.set_groundtruth_dict()

    def predict(self, input_data):
        outputs_dict = self._decoder.predict(input_data)

        return outputs_dict

    def loss(self, input_data):
        loss = self._decoder.loss(input_data)
        return loss

    def sequence_dist(self, input_data):
        seq_dis = self._decoder.sequence_dist(input_data)

        return seq_dis


class BidirectionSTS(object):
    """
        Implement the sts decoder for sequence recognition, which only has one decoding mode (from left to right).
    """

    def __init__(self, phase, sequence_length):
        """
        :param phase: 'Train' or 'Test'
        :param sequence_length:
        """
        self._normal_decoder = STSDecoder(phase, sequence_length, 'Normal')
        self._reverse_decoder = STSDecoder(phase, sequence_length, 'Reverse')

    def set_label(self, label):
        self._normal_decoder.set_label(label['LeftToRight'])
        self._normal_decoder.set_groundtruth_dict()

        self._reverse_decoder.set_label(label['RightToLeft'])
        self._reverse_decoder.set_groundtruth_dict()

    def predict(self, input_data):
        outputs_dict = dict()
        normal_outputs_dict = self._normal_decoder.predict(input_data)
        reverse_outputs_dict = self._reverse_decoder.predict(input_data)

        outputs_dict['normal'] = normal_outputs_dict
        outputs_dict['reverse'] = reverse_outputs_dict

        return outputs_dict

    def loss(self, input_data):
        normal_loss = self._normal_decoder.loss(input_data['normal'])
        reverse_loss = self._reverse_decoder.loss(input_data['reverse'])

        total_loss = list()
        total_loss.append(normal_loss)
        total_loss.append(reverse_loss)
        with tf.name_scope('Combine_Loss'):
            total_loss = tf.reduce_mean(total_loss)

        return total_loss

    def sequence_dist(self, input_data):
        normal_seq_dis = self._normal_decoder.sequence_dist(input_data['normal'])
        reverse_seq_dis = self._reverse_decoder.sequence_dist(input_data['reverse'])

        seq_dis = list()
        seq_dis.append(normal_seq_dis)
        seq_dis.append(reverse_seq_dis)
        with tf.name_scope('Combine_Seq_Dis'):
            seq_dis = tf.reduce_mean(seq_dis)

        return seq_dis
