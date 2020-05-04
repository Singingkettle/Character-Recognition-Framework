#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: test.py
@time: 2019/7/23 15:05
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import argparse
import os.path as ops

import cv2
import glog as logger
import matplotlib.pyplot as plt
import tensorflow as tf

from config.config import config
from config.config import update_config
from data_provider import tf_io_pipline_fast_tools
from decoder.decoder import get_decoder
from encoder.encoder import get_encoder


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


def init_args():
    """

    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser(description='Test single image')
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default='/home/citybuster/Projects/CRNN_STN_SEQ/experiments'
                '/synth90k/shadownet/bidirection_sts_shadownet_sgd_exponential-lr.yaml',
        type=str
    )
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--image_path', type=str,
                        help='Path to the image to be tested',
                        default='/home/citybuster/Projects/CRNN_STN_SEQ/data/test_images/test_01.jpg')
    parser.add_argument('--weights_path', type=str,
                        help='Path to the model file',
                        default='/home/citybuster/Projects/CRNN_STN_SEQ/output/mjsynth/'
                                'TextRecognition_mjsynth/bidirection_sts_shadownet_sgd_exponential-lr.yaml/'
                                'TextRecognition_2019-07-24-20-48-50.ckpt-2000000')
    parser.add_argument('-v', '--visualize', type=bool,
                        help='Whether to display images',
                        default=False)

    return parser.parse_args()


def recognize(image_path, weights_path, is_vis):
    """

    :param image_path:
    :param weights_path:
    :param is_vis:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if config.USE_STN:
        new_heigth = config.STN.IH
        new_width = config.STN.IW
        image = cv2.resize(image, dsize=(config.STN.IW, config.STN.IH), interpolation=cv2.INTER_LINEAR)
    else:
        new_heigth = config.ENCODER.IH
        new_width = config.ENCODER.IW
        image = cv2.resize(image, dsize=(config.ENCODER.IW, config.ENCODER.IH), interpolation=cv2.INTER_LINEAR)
    image_vis = image

    # Set up data placeholder
    char_dict_path = config.CHAR_DICT
    ord_map_dict_path = config.ORD_MAP_DICT
    inputdata = tf.placeholder(
        dtype=tf.float32,
        shape=[1, new_heigth, new_width, config.ENCODER.INPUT_CHANNELS],
        name='input'
    )

    # Set up convert
    convert = tf_io_pipline_fast_tools.FeatureReader(
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path
    )

    # Set up network graph
    encoder_net = get_encoder('test')
    decoder_net = get_decoder('test')

    gpu_list = config.GPUS
    gpu_list = gpu_list.split(',')

    device_name = '/gpu:{}'.format(gpu_list[0])
    with tf.device(device_name):
        with tf.name_scope('Test'):
            encoder_out = encoder_net.forward(inputdata, name=config.ENCODER.NETWORKTYPE + 'Encoder')
            decoder_out = decoder_net.predict(encoder_out)

    # Config tf saver
    saver = tf.train.Saver()

    # config tf session
    sess_config = tf.ConfigProto(allow_soft_placement=True)  # 允许tf自动选择一个存在并且可用的设备来运行操作
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.TEST.TF_ALLOW_GROWTH

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        decoder_out_value = sess.run(decoder_out, feed_dict={inputdata: [image]})

        if config.DECODER_MODEL in config.VALID_DECODER_MODEL:
            if config.DECODER_MODEL == 'normal_ctc' or config.DECODER_MODEL == 'normal_sts':
                preds = convert.sparse_tensor_to_str(decoder_out_value['labels'])
            elif config.DECODER_MODEL == 'reverse_sts':
                preds = convert.sparse_tensor_to_str(decoder_out_value['labels'], reverse=True)
            elif config.DECODER_MODEL == 'bidirection_sts':
                normal_preds = convert.sparse_tensor_to_str(decoder_out_value['normal']['labels'])
                reverse_preds = convert.sparse_tensor_to_str(decoder_out_value['reverse']['labels'], reverse=True)
                if decoder_out_value['normal']['scores'] > decoder_out_value['reverse']['scores']:
                    preds = normal_preds
                else:
                    preds = reverse_preds
            else:
                raise ValueError('Unknown decoder model: {}'.format(config.DECODER_MODEL))
        else:
            raise ValueError('Unknown decoder model: {}'.format(config.DECODER_MODEL))

        logger.info('Predict image {:s} result {:s}'.format(ops.split(image_path)[1], preds[0]))

        if is_vis:
            plt.figure('CRNN Model Demo')
            plt.imshow(image_vis)
            plt.show()

    sess.close()

    return


if __name__ == '__main__':
    # init images
    args = init_args()

    # detect images
    recognize(image_path=args.image_path, weights_path=args.weights_path, is_vis=args.visualize)
