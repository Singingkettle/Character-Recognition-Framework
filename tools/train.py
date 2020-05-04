#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: train.py
@time: 2019/6/18 21:21
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import _init_paths
import argparse
import os
import pprint
import time

import math
import numpy as np
import tensorflow as tf

from config.config import config
from config.config import update_config
from data_provider import tf_io_pipline_fast_tools, data
from decoder.decoder import get_decoder
from encoder.encoder import get_encoder
from local_utils.utils import create_logger, average_gradients, compute_avg_accuracy
from optimizer import optimizer_builder


def parse_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser(description='Train text recognition network')
    # general
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default='/home/citybuster/Projects/CRNN_STN_SEQ/experiments'
                '/synth90k/shadownet/normal_sts_shadownet_sgd_exponential-lr.yaml',
        type=str
    )

    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)
    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)

    args = parser.parse_args()

    return args


def reset_config(args):
    if args.gpus:
        config.GPUS = args.gpus


def compute_net_gradients(images, labels, encoder_net, decoder_net, optimizer=None):
    """
    Calculate gradients for single GPU
    :param images: images for training
    :param labels: labels corresponding to images
    :param encoder_net:
    :param decoder_net:
    :param optimizer: network optimizer
    :return:
    """

    encoder_out = encoder_net.forward(images, name=config.ENCODER.NETWORKTYPE + 'Encoder')
    decoder_net.set_label(labels)
    decoder_out = decoder_net.predict(encoder_out)
    decoder_loss = decoder_net.loss(decoder_out)

    if optimizer is not None:
        grads = optimizer.compute_gradients(decoder_loss)
    else:
        grads = None

    return decoder_loss, grads


def train_single(logger, tb_log_dir):
    """
    :param logger:
    :param tb_log_dir:
    :return:
    """
    # prepare dataset
    if config.USE_STN:
        tfrecords_dir = config.DATA_DIR + '_stn'
    else:
        tfrecords_dir = config.DATA_DIR

    # Set up summary writer
    global_summaries = set([])
    summary_writer = tf.summary.FileWriter(tb_log_dir)

    # Set up data provider
    char_dict_path = config.CHAR_DICT
    ord_map_dict_path = config.ORD_MAP_DICT
    train_images, train_labels, train_images_paths, \
    val_images, val_labels, val_images_paths = data.get_data(tfrecords_dir, char_dict_path, ord_map_dict_path)

    # Set up convert
    convert = tf_io_pipline_fast_tools.FeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    # Set up network graph
    train_encoder_net = get_encoder('train')
    train_decoder_net = get_decoder('train')

    val_encoder_net = get_encoder('val')
    val_decoder_net = get_decoder('val')

    gpu_list = config.GPUS
    gpu_list = gpu_list.split(',')

    device_name = '/gpu:{}'.format(gpu_list[0])
    with tf.device(device_name):
        with tf.name_scope('Train') as train_scope:
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                train_encoder_out = train_encoder_net.forward(train_images, name=config.ENCODER.NETWORKTYPE + 'Encoder')
                train_decoder_net.set_label(train_labels)
                train_decoder_out = train_decoder_net.predict(train_encoder_out)
                train_loss = train_decoder_net.loss(train_decoder_out)
                train_sequence_dist = train_decoder_net.sequence_dist(train_decoder_out)

        with tf.name_scope('Val') as _:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                val_encoder_out = val_encoder_net.forward(val_images, name=config.ENCODER.NETWORKTYPE + 'Encoder')
                val_decoder_net.set_label(val_labels)
                val_decoder_out = val_decoder_net.predict(val_encoder_out)
                val_loss = val_decoder_net.loss(val_decoder_out)
                val_sequence_dist = val_decoder_net.sequence_dist(val_decoder_out)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # set optimizer
            optimizer = optimizer_builder.build(global_summaries)
            optimizer = optimizer.minimize(loss=train_loss, global_step=tf.train.get_or_create_global_step())

    # Gather initial summaries.
    global_summaries.add(tf.summary.scalar(name='train_loss', tensor=train_loss))
    global_summaries.add(tf.summary.scalar(name='val_loss', tensor=val_loss))
    global_summaries.add(tf.summary.scalar(name='train_seq_distance', tensor=train_sequence_dist))
    global_summaries.add(tf.summary.scalar(name='val_seq_distance', tensor=val_sequence_dist))

    # Set saver configuration
    saver = tf.train.Saver()
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = '{:s}_{:s}.ckpt'.format(config.MODEL.NAME, str(train_start_time))
    model_save_path = os.path.join(config.MODEL_SAVE_DIR, model_name)

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.TRAIN.TF_ALLOW_GROWTH

    #
    sess = tf.Session(config=sess_config)

    # Merge all summaries together.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES, train_scope))
    summaries |= global_summaries
    merge_summary_op = tf.summary.merge(list(summaries), name='summary_op')

    summary_writer.add_graph(sess.graph)
    # Set the training parameters
    train_epochs = config.TRAIN.EPOCHS

    with sess.as_default():
        epoch = 0
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/single_model.pb'.format(config.MODEL_SAVE_DIR))
        if not config.RESUME_PATH:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(config.RESUME_PATH))
            saver.restore(sess=sess, save_path=config.TRAIN.RESUME)
            epoch = sess.run(tf.train.get_global_step())

        patience_counter = 1
        cost_history = [np.inf]
        while epoch < train_epochs:
            epoch += 1
            # setup early stopping
            if epoch > 1 and config.TRAIN.EARLY_STOPPING:
                # We always compare to the first point where cost didn't improve
                if cost_history[-1 - patience_counter] - cost_history[-1] > config.TRAIN.PATIENCE_DELTA:
                    patience_counter = 1
                else:
                    patience_counter += 1
                if patience_counter > config.TRAIN.PATIENCE_EPOCHS:
                    logger.info("Cost didn't improve beyond {:f} for {:d} epochs, stopping early.".
                                format(config.TRAIN.PATIENCE_DELTA, patience_counter))
                    break

            if config.TRAIN.DECODE and epoch % 500 == 0:
                # train part
                _, train_loss_value, train_decoder_out_dict, train_seq_dist_value, train_labels_sparse, \
                merge_summary_value = sess.run(
                    [optimizer, train_loss, train_decoder_out, train_sequence_dist, train_labels, merge_summary_op])

                avg_train_accuracy = compute_avg_accuracy(convert, train_labels_sparse, train_decoder_out_dict)
                if epoch % config.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                        epoch + 1, train_loss_value, train_seq_dist_value, avg_train_accuracy))

                # validation part
                val_loss_value, val_decoder_out_dict, val_seq_dist_value, val_labels_sparse = sess.run(
                    [val_loss, val_decoder_out, val_sequence_dist, val_labels])

                avg_val_accuracy = compute_avg_accuracy(convert, val_labels_sparse, val_decoder_out_dict)
                if epoch % config.TRAIN.VAL_DISPLAY_STEP == 0:
                    logger.info('Epoch_Val: {:d} cost= {:9f} seq distance= {:9f} val accuracy= {:9f}'.format(
                        epoch + 1, val_loss_value, val_seq_dist_value, avg_val_accuracy))

                summary_fly = tf.Summary(value=[tf.Summary.Value(tag='acc_train', simple_value=avg_train_accuracy),
                                                tf.Summary.Value(tag='acc_val', simple_value=avg_val_accuracy),
                                                ])
                summary_writer.add_summary(summary=summary_fly, global_step=epoch)
            else:
                _, train_loss_value, merge_summary_value = sess.run([optimizer, train_loss, merge_summary_op])

                if epoch % config.TRAIN.DISPLAY_STEP == 0:
                    logger.info('Epoch_Train: {:d} cost= {:9f}'.format(epoch + 1, train_loss_value))

            # record history train ctc loss
            cost_history.append(train_loss_value)
            # add training sumary
            summary_writer.add_summary(summary=merge_summary_value, global_step=epoch)

            if epoch % 2000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    return np.array(cost_history[1:])  # Don't return the first np.inf


def train_multi(logger, tb_log_dir):
    """
    :param logger:
    :param tb_log_dir:
    :return:
    """
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # prepare dataset
        if config.USE_STN:
            tfrecords_dir = config.DATA_DIR + '_stn'
        else:
            tfrecords_dir = config.DATA_DIR

        # Set up summary writer
        global_summaries = set([])
        summary_writer = tf.summary.FileWriter(tb_log_dir)

        # Set up data provider
        char_dict_path = config.CHAR_DICT
        ord_map_dict_path = config.ORD_MAP_DICT
        train_images, train_labels, train_images_paths, \
        val_images, val_labels, val_images_paths = data.get_data(tfrecords_dir, char_dict_path, ord_map_dict_path)

        # Set up network graph
        train_encoder_net = get_encoder('train')
        train_decoder_net = get_decoder('train')

        val_encoder_net = get_encoder('val')
        val_decoder_net = get_decoder('val')

        # set average container
        tower_grads = []
        train_scopes = []
        train_tower_loss = []
        val_tower_loss = []
        batchnorm_updates = None

        # Set up optimizer
        optimizer = optimizer_builder.build(global_summaries)

        gpu_list = config.GPUS
        gpu_list = gpu_list.split(',')
        # set distributed train op

        is_network_initialized = False
        for i, gpu_id in enumerate(gpu_list):
            with tf.device('/gpu:{}'.format(gpu_id)):
                with tf.name_scope('Train_{:d}'.format(i)) as train_scope:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=is_network_initialized):
                        train_loss, grads = compute_net_gradients(train_images, train_labels,
                                                                  train_encoder_net, train_decoder_net, optimizer)

                        is_network_initialized = True
                        train_scopes.append(train_scope)

                        # Only use the mean and var in the first gpu tower to update the parameter
                        if i == 0:
                            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                        tower_grads.append(grads)
                        train_tower_loss.append(train_loss)
                with tf.name_scope('Val_{:d}'.format(i)) as _:
                    with tf.variable_scope(tf.get_variable_scope(), reuse=is_network_initialized):
                        val_loss, _ = compute_net_gradients(val_images, val_labels, val_encoder_net, val_decoder_net)
                        val_tower_loss.append(val_loss)

        with tf.name_scope('Average_Grad'):
            grads = average_gradients(tower_grads)
        with tf.name_scope('Average_Loss'):
            avg_train_loss = tf.reduce_mean(train_tower_loss)
            avg_val_loss = tf.reduce_mean(val_tower_loss)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            global_summaries.add(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables
        variable_averages = tf.train.ExponentialMovingAverage(config.TRAIN.MOVING_AVERAGE_DECAY,
                                                              num_updates=tf.train.get_or_create_global_step())
        variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
        variables_averages_op = variable_averages.apply(variables_to_average)

        # Group all the op needed for training
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=tf.train.get_or_create_global_step())
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

        global_summaries.add(tf.summary.scalar(name='average_train_loss', tensor=avg_train_loss))

        # Merge all summaries together.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES, train_scopes[0]))
        summaries |= global_summaries
        train_merge_summary_op = tf.summary.merge(list(summaries), name='train_summary_op')
        val_merge_summary_op = tf.summary.merge([tf.summary.scalar(name='average_val_loss', tensor=avg_val_loss)],
                                                name='val_summary_op')

        # Set saver configuration
        saver = tf.train.Saver()
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = '{:s}_{:s}.ckpt'.format(config.MODEL.NAME, str(train_start_time))
        model_save_path = os.path.join(config.MODEL_SAVE_DIR, model_name)

        # set sess config
        sess_config = tf.ConfigProto(device_count={'GPU': len(gpu_list)}, allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = config.TRAIN.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = config.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        # Set the training parameters
        train_epochs = config.TRAIN.EPOCHS

        logger.info('Global configuration is as follows:')
        logger.info(config)

        sess = tf.Session(config=sess_config)

        summary_writer.add_graph(sess.graph)

        epoch = 0
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/multi_model.pb'.format(config.MODEL_SAVE_DIR))
        if not config.RESUME_PATH:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(config.RESUME_PATH))
            saver.restore(sess=sess, save_path=config.TRAIN.RESUME)
            epoch = sess.run(tf.train.get_global_step())

        train_cost_time_mean = []
        val_cost_time_mean = []

        while epoch < train_epochs:
            epoch += 1
            # training part
            t_start = time.time()

            _, train_loss_value, train_summary = sess.run(fetches=[train_op, avg_train_loss, train_merge_summary_op])

            if math.isnan(train_loss_value):
                raise ValueError('Train loss is nan')

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            t_start_val = time.time()

            val_loss_value, val_summary = sess.run(fetches=[avg_val_loss, val_merge_summary_op])
            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            summary_writer.add_summary(val_summary, global_step=epoch)

            if epoch % config.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch_Train: {:d} total_loss= {:6f} mean_cost_time= {:5f}s '.
                            format(epoch + 1, train_loss_value, np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % config.TRAIN.VAL_DISPLAY_STEP == 0:
                logger.info('Epoch_Val: {:d} total_loss= {:6f} mean_cost_time= {:5f}s '.
                            format(epoch + 1, val_loss_value, np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 5000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
        sess.close()

    return


def main():
    args = parse_args()
    reset_config(args)
    logger, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    gpu_list = config.GPUS
    gpu_list = gpu_list.split(',')

    if len(gpu_list) > 1:
        logger.info('Use multi gpu to train the model')
        train_multi(logger, tb_log_dir)
    else:
        logger.info('Use single gpu to train the model')
        train_single(logger, tb_log_dir)


if __name__ == '__main__':
    main()
