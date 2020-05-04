#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: utils.py
@time: 2019/7/23 14:15
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import argparse
import logging
import os
import time
from logging import handlers
from pathlib import Path
from local_utils import evaluation_tools
import tensorflow as tf

from config.config import get_model_name, config


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.NAME
    model = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('/')[-1]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    log = Logger(str(final_log_file), level='info')

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # Set the model save dir
    config.MODEL_SAVE_DIR = str(final_output_dir)

    return log.logger, str(tensorboard_log_dir)


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def compute_net_gradients(images, labels, net, optimizer=None, is_net_first_initialized=False):
    """
    Calculate gradients for single GPU
    :param images: images for training
    :param labels: labels corresponding to images
    :param net: classification model
    :param optimizer: network optimizer
    :param is_net_first_initialized: if the network is initialized
    :return:
    """
    _, net_loss = net.compute_loss(
        inputdata=images,
        labels=labels,
        name='shadow_net',
        reuse=is_net_first_initialized
    )

    if optimizer is not None:
        grads = optimizer.compute_gradients(net_loss)
    else:
        grads = None

    return net_loss, grads


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_shape[index])

    return combined_shape


def compute_accuracy(convert, ground_truth_labels, prediction_labels):

    ground_truth_str = convert.sparse_tensor_to_str(ground_truth_labels)
    predictions_str = convert.sparse_tensor_to_str(prediction_labels)
    accuracy = evaluation_tools.compute_accuracy(ground_truth_str, predictions_str, mode='full_sequence')

    return accuracy


def compute_avg_accuracy(convert, sparse_labels, decoder_out_dict):

    if config.DECODER_MODEL in config.VALID_DECODER_MODEL:
        accuracy = None
        if config.DECODER_MODEL == 'normal_ctc' or config.DECODER_MODEL == 'normal_sts':
            accuracy = compute_accuracy(convert, sparse_labels['LeftToRight'], decoder_out_dict['labels'])
        elif config.DECODER_MODEL == 'reverse_sts':
            accuracy = compute_accuracy(convert, sparse_labels['RightToLeft'], decoder_out_dict['labels'])
        elif config.DECODER_MODEL == 'bidirection_sts':
            normal_accuracy = compute_accuracy(convert, sparse_labels['LeftToRight'],
                                               decoder_out_dict['normal']['labels'])
            reverse_accuracy = compute_accuracy(convert, sparse_labels['RightToLeft'],
                                                decoder_out_dict['reverse']['labels'])
            accuracy = (normal_accuracy + reverse_accuracy) / 2
        else:
            raise ValueError('Unknown decoder model: {}'.format(config.DECODER_MODEL))
        return accuracy
    else:
        raise ValueError('Unknown decoder model: {}'.format(config.DECODER_MODEL))

