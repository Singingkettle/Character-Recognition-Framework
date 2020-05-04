#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: decoder.py
@time: 2019/7/22 17:19
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import numpy as np

from config.config import config
from decoder.decoder_ctc import NormalCTC
from decoder.decoder_seqtoseq import NormalSTS, ReverseSTS, BidirectionSTS

decoder_factory = {
    'normal_ctc': NormalCTC,
    'normal_sts': NormalSTS,
    'reverse_sts': ReverseSTS,
    'bidirection_sts': BidirectionSTS
}


def get_decoder(phase):
    decoder = config.DECODER_MODEL
    if decoder in config.VALID_DECODER_MODEL:
        decoder = decoder_factory[decoder]
        if phase == 'train':
            batch_size = config.TRAIN.BATCH_SIZE
        else:
            batch_size = 1
        decoder_net = decoder(phase, config.DATASET.SEQ_LENGTH * np.ones(batch_size))
        return decoder_net
    else:
        raise ValueError('Unknown decoder model: {}'.format(decoder))
