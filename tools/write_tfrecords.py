"""
Write tfrecords tools
"""
import _init_paths
import argparse
import pprint
import os.path as ops
import os
import glog as logger
from data_provider import data_feed_pipline
from config.config import config
from config.config import update_config


def parse_args():
    """
    :return: parsed arguments and (updated) config.cfg object
    """
    parser = argparse.ArgumentParser(description='Train text recognition network')
    # general
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default='/home/citybuster/Projects/CRNN_STN_SEQ/experiments/'
                'synth90k/shadownet/stn_ctc_shadownet_sgd_exponential-lr.yaml',
        type=str
    )

    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)
    parser.add_argument('--data',
                        help='The raw dataset dir',
                        default='/home/citybuster/Data/mnt/ramdisk/max/90kDICT32px',
                        type=str)

    args = parser.parse_args()
    return args


def write_tfrecords(dataset_dir):
    """
    Write tensorflow records for training , testing and validation
    :param dataset_dir: the root dir of crnn dataset
    :return:
    """
    assert ops.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)
    if config.USE_STN:
        save_dir = config.DATA_DIR + '_stn'
    else:
        save_dir = config.DATA_DIR
    char_dict_path = config.CHAR_DICT
    ord_map_dict_path = config.ORD_MAP_DICT
    os.makedirs(save_dir, exist_ok=True)

    # test data producer
    producer = data_feed_pipline.DataProducer(
        dataset_dir=dataset_dir,
        char_dict_path=char_dict_path,
        ord_map_dict_path=ord_map_dict_path,
        writer_process_nums=30
    )

    producer.generate_tfrecords(
        save_dir=save_dir
    )


def main():
    args = parse_args()
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    dataset_dir = args.data
    write_tfrecords(dataset_dir)


if __name__ == '__main__':
    """
    generate tfrecords
    """
    main()
