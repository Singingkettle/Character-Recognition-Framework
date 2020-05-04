#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: optimizer_builder.py
@time: 2019/7/17 20:36
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import tensorflow as tf

from config.config import config
from optimizer import learning_schedules


def _create_learning_rate(learning_rate_type, global_summaries):
    """Create optimizer learning rate based on config.
    Args:
        learning_rate_type：
        global_summaries: A set to attach learning rate summary to.
    Returns:
        A learning rate.
    Raises:
      ValueError: when using an unsupported input data type.
    """
    learning_rate = None
    if learning_rate_type == 'constant_learning_rate':
        learning_rate = config.LRC.LEARNING_RATE

    if learning_rate_type == 'exponential_decay_learning_rate':
        learning_rate = tf.train.exponential_decay(
            config.LRE.LEARNING_RATE,
            tf.train.get_or_create_global_step(),
            config.LRE.DECAY_STEPS,
            config.LRE.DECAY_RATE,
            staircase=config.LRE.STAIRCASE)

    if learning_rate_type == 'manual_step_learning_rate':
        if len(config.LRM.STEPS) == len(config.LRM.LEARNING_RATES):
            raise ValueError('Empty learning rate schedule or The size of steps and learning_rate is not match')
        learning_rate_step_boundaries = config.LRM.DECAY_STEPS
        learning_rate_sequence = [config.initial_learning_rate]
        learning_rate_sequence += config.LRM.LEARNING_RATES
        learning_rate = learning_schedules.manual_stepping(
            tf.train.get_or_create_global_step(), learning_rate_step_boundaries,
            learning_rate_sequence)

    if learning_rate is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

    global_summaries.add(tf.summary.scalar('LearningRate', learning_rate))
    return learning_rate


def build(global_summaries):
    """Create optimizer based on config.

    Args:
      global_summaries: A set to attach learning rate summary to.

    Returns:
      An optimizer.

    Raises:
      ValueError: when using an unsupported input data type.
    """
    optimizer_type = config.TRAIN.OPTIMIZER
    optimizer = None

    if optimizer_type == 'rms_prop_optimizer':
        optimizer = tf.train.RMSPropOptimizer(
            _create_learning_rate(config.RMSProp_OPTIMIZER.LEARNING_RATE, global_summaries),
            decay=config.RMSProp_OPTIMIZER.decay,
            momentum=config.RMSProp_OPTIMIZER.MOMENTMU,
            epsilon=config.RMSProp_OPTIMIZER.EPSILON)

    if optimizer_type == 'momentum_optimizer':
        optimizer = tf.train.MomentumOptimizer(
            _create_learning_rate(config.Momentum_OPTIMIZER.LEARNING_RATE_TYPE, global_summaries),
            momentum=config.Momentum_OPTIMIZER.MOMENTUM)

    if optimizer_type == 'adam_optimizer':
        optimizer = tf.train.AdamOptimizer(
            _create_learning_rate(config.Adam_OPTIMIZER.LEARNING_RATE_TYPE, global_summaries))

    if optimizer_type == 'nadam_optimizer':
        optimizer = tf.contrib.opt.NadamOptimizer(
            _create_learning_rate(config.Nadam_OPTIMIZER.LEARNING_RATE_TYPE, global_summaries))

    if optimizer_type == 'adadelta_optimizer':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=_create_learning_rate(config.Adadelta_OPTIMIZER.LEARNING_RATE_TYPE, global_summaries),
            rho=config.rho)

    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)

    if config.TRAIN.USE_MOVING_AVERAGE:
        optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer, average_decay=config.TRAIN.MOVING_AVERAGE_DECAY)

    return optimizer
