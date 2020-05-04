#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: data.py
@time: 2019/6/18 10:43
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import tensorflow as tf

from config.config import config
from data_provider import data_feed_pipline


def get_data(tfrecords_dir, char_dict_path, ord_map_dict_path):
    with tf.name_scope('Train_Data'):
        train_dataset = data_feed_pipline.DataFeeder(
            tfrecords_dir=tfrecords_dir,
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path,
            flags='train'
        )
        train_images, train_flabels, train_blabels, train_images_paths = train_dataset.inputs(
            batch_size=config.TRAIN.BATCH_SIZE
        )

    with tf.name_scope('Val_Data'):
        val_dataset = data_feed_pipline.DataFeeder(
            tfrecords_dir=tfrecords_dir,
            char_dict_path=char_dict_path,
            ord_map_dict_path=ord_map_dict_path,
            flags='val'
        )
        val_images, val_flabels, val_blabels, val_images_paths = val_dataset.inputs(
            batch_size=config.TRAIN.BATCH_SIZE
        )

    train_labels = dict()
    train_labels['LeftToRight'] = train_flabels
    train_labels['RightToLeft'] = train_blabels

    val_labels = dict()
    val_labels['LeftToRight'] = val_flabels
    val_labels['RightToLeft'] = val_blabels

    return train_images, train_labels, train_images_paths, val_images, val_labels, val_images_paths
