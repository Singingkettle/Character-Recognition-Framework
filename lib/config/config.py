#!/usr/bin/python3
# encoding: utf-8
"""
@author: ShuoChang
@license: (C) MIT.
@contact: changshuo@bupt.edu.cn
@software: CRNN_STN_SEQ
@file: config.py
@time: 2019/6/22 17:20
@blog: https://www.zhihu.com/people/chang-shuo-59/activities
"""

import os

import yaml
from easydict import EasyDict as edict

config = edict()
# ======================================================================================================================
# Global settings
config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = ''
config.CHAR_DICT = ''
config.ORD_MAP_DICT = ''
config.PRINT_FREQ = 20
config.USE_STN = False
config.DECODER_MODEL = 'normal_ctc'  # There are four modes: normal_ctc, normal_sts, reverse_sts, bidirection_sts
config.VALID_DECODER_MODEL = ['normal_ctc', 'normal_sts', 'reverse_sts', 'bidirection_sts']
config.MODEL_SAVE_DIR = None  # This variable will be set on the fly
config.RESUME_PATH = None  # Restore training from the check point

# ======================================================================================================================
config.MODEL = edict()
config.MODEL.NAME = 'TextRecognition'
config.MODEL.PRETRAINED = ''

# ======================================================================================================================
config.DATASET = edict()
config.DATASET.NAME = 'mjsynth'
# Number character classes
config.DATASET.NUM_CLASSES = 37  # synth90k dataset
# Sequence length.  This has to be the width of the final feature map of the CNN, which is input size width / 4
config.DATASET.SEQ_LENGTH = 25  # synth90k dataset

# ======================================================================================================================
config.STN = edict()
# Set the input image size of stn
config.STN.IH = 64
config.STN.IW = 256
# Set the input localization image size of stn
config.STN.ILH = 32
config.STN.ILW = 64
# Set the output image size of stn
config.STN.OH = 32
config.STN.OW = 100
# Set the clip margin of stn
config.STN.MX = 0
config.STN.MY = 0
# Set the number of control points
config.STN.NCP = 20
# Set the init bias pattern of stn
config.STN.IB = 'sine'

# Set the conv regularizer in stn
config.STN.CONVR = 'l2_regularizer'
# Set the conv initializer in stn
config.STN.CONVI = 'variance_scaling_initializer'
# Set the batch norm decay of conv in stn
config.STN.CBND = 0.99
# Set the fc regularizer in stn
config.STN.FCR = 'l2_regularizer'
# Set the fc initializer in stn
config.STN.FCI = 'variance_scaling_initializer'
# Set the batch norm decay of fc in stn
config.STN.FCBND = 0.99
# Set the summary activation of the stn
config.STN.SUMMARY_ACTIVATION = True

# ======================================================================================================================
# Set the encoder network
config.ENCODER = edict()
# The cnn network to extract features
config.ENCODER.NETWORKTYPE = 'shadow'
# Number of units in each LSTM cell
config.ENCODER.HIDDEN_UNITS = 256
# Number of stacked LSTM cells
config.ENCODER.HIDDEN_LAYERS = 2
# Width x height into which training / testing images are resized before feeding into the network
config.ENCODER.IW = 100
config.ENCODER.IH = 32  # synth90k dataset, which is the same as the output image size of stn when use stn
# Number of channels in images
config.ENCODER.INPUT_CHANNELS = 3
# Set the summary activation of the encoder
config.ENCODER.SUMMARY_ACTIVATION = True

# ======================================================================================================================
# Set decoder CTC network
config.DECODER_CTC = edict()
# Set the beam search width in CTC
config.DECODER_CTC.BW = 5

# ======================================================================================================================
# Set the attentional seq-to-seq decoder network (STS)
config.DECODER_STS = edict()
# Set the num units of rnn cell in STS
config.DECODER_STS.NU = 256
# Set the forget bias of rnn cell in STS
config.DECODER_STS.FB = 1.0

# Set the initializer of rnn cell in STS
config.DECODER_STS.I = 'orthogonal_initializer'
# Set TruncatedNormalInitializer mean
config.DECODER_STS.TM = 0.0
# Set TruncatedNormalInitializer stddev
config.DECODER_STS.TS = 1.0
# Set OrthogonalInitializer gain
config.DECODER_STS.OG = 0.0
# Set OrthogonalInitializer seed
config.DECODER_STS.OS = 0.0
# Set UniformInitializer minval
config.DECODER_STS.UMI = 0.0
# Set UniformInitializer maxval
config.DECODER_STS.UMA = 1.0

# Set the regularizer of STS
config.DECODER_STS.R = 'l2_regularizer'
# Set the weight of L1 regularizer
config.DECODER_STS.L1W = 0.0001
# Set the weight of L1 regularizer
config.DECODER_STS.L2W = 0.0001
# Set the syc to choose which kind of seqtoseq wrapper to use
config.DECODER_STS.SYNC = True
# Set the beam width of STS
config.DECODER_STS.BW = 5
# Set the max num steps of STS
config.DECODER_STS.MNS = 30

# If decode in two directions (only for SeqToSeq framework)
config.DECODER_STS.REVERSE = True

# ======================================================================================================================
# Train options
config.TRAIN = edict()

# Use early stopping?
config.TRAIN.EARLY_STOPPING = False
# Wait at least this many epochs without improvement in the cost function
config.TRAIN.PATIENCE_EPOCHS = 6
# Expect at least this improvement in one epoch in order to reset the early stopping counter
config.TRAIN.PATIENCE_DELTA = 1e-3
# Set the shadownet training epochs
config.TRAIN.EPOCHS = 2000000
# Set the display step
config.TRAIN.DISPLAY_STEP = 100
# Set the GPU resource used during training process
config.TRAIN.GPU_MEMORY_FRACTION = 0.9
# Set the GPU allow growth parameter during tensorflow training process
config.TRAIN.TF_ALLOW_GROWTH = True
# Set the training batch size
config.TRAIN.BATCH_SIZE = 32
# Set the validation batch size
config.TRAIN.VAL_BATCH_SIZE = 32
# Set multi process nums
config.TRAIN.CPU_MULTI_PROCESS_NUMS = 6
# Set val display step
config.TRAIN.VAL_DISPLAY_STEP = 1000
# Set the type of optimizer
config.TRAIN.OPTIMIZER = 'momentum_optimizer'
# Use moving average of optimizer
config.TRAIN.USE_MOVING_AVERAGE = False
# Set moving average decay
config.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
# Activate decoding of predictions during training (slow!)
config.TRAIN.DECODE = False

# **********************************************************************************************************************
# RMSProp optimizer options
config.RMSProp_OPTIMIZER = edict()

# Set the decay parameter of the RMSProp optimizer
config.RMSProp_OPTIMIZER.DECAY = 0.9
# Set the momentum parameter of the RMSProp optimizer
config.RMSProp_OPTIMIZER.MOMENTUM = 0.9
# Set the learning rate type of the RMSProp optimizer
config.RMSProp_OPTIMIZER.LEARNING_RATE_TYPE = 'exponential_decay_learning_rate'
# Set the epsilon parameter of the RMSProp optimizer
config.RMSProp_OPTIMIZER.EPSILON = 1.0

# **********************************************************************************************************************
# Momentum optimizer options
config.Momentum_OPTIMIZER = edict()

# Set the momentum parameter of the Momentum optimizer
config.Momentum_OPTIMIZER.MOMENTUM = 0.9
# Set the learning rate type of the Momentum optimizer
config.Momentum_OPTIMIZER.LEARNING_RATE_TYPE = 'exponential_decay_learning_rate'

# **********************************************************************************************************************
# Adam optimizer options
config.Adam_OPTIMIZER = edict()

# Set the learning rate type of the Adam optimizer
config.Adam_OPTIMIZER.LEARNING_RATE_TYPE = 'exponential_decay_learning_rate'

# **********************************************************************************************************************
# Nadam optimizer options
config.Nadam_OPTIMIZER = edict()

# Set the learning rate type of the Nadam optimizer
config.Nadam_OPTIMIZER.LEARNING_RATE_TYPE = 'exponential_decay_learning_rate'

# **********************************************************************************************************************
# Adadelta optimizer options
config.Adadelta_OPTIMIZER = edict()

# Set the learning rate type of the Adadelta optimizer
config.Adadelta_OPTIMIZER.LEARNING_RATE_TYPE = 'exponential_decay_learning_rate'
# Set the rho parameter of the Adadelta optimizer
config.Adadelta_OPTIMIZER.RHO = 0.95

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Exponential decay learning rate
config.LRC = edict()

# Set the initial learning rate in constant mode
config.LRC.LEARNING_RATE = 0.01

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Exponential decay learning rate
config.LRE = edict()

# Set the initial learning rate of the exponential_decay mode
config.LRE.LEARNING_RATE = 0.01
# Set the decay steps of the exponential_decay mode
config.LRE.DECAY_STEPS = 500000
# Set the decay rate of the exponential_decay mode
config.LRE.DECAY_RATE = 0.1
# Set the staircase of the exponential_decay mode
config.LRE.STAIRCASE = True

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Manual step learning rate
config.LRM = edict()

# Set the initial learning rate of the manual mode
config.LRM.LEARNING_RATE = 0.01
# Set the decay steps of the manual mode
config.LRM.DECAY_STEPS = [600000, 800000, 100000]
# Set the multi-stage learning rates of the manual mode
config.LRM.LEARNING_RATES = [1e-1, 1e-2, 1e-3]

# ======================================================================================================================
# Test options
config.TEST = edict()

# Set the GPU resource used during testing process
config.TEST.GPU_MEMORY_FRACTION = 0.6
# Set the GPU allow growth parameter during tensorflow testing process
config.TEST.TF_ALLOW_GROWTH = False
# Set the test batch size
config.TEST.BATCH_SIZE = 32


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

    gpus = [int(i) for i in config.GPUS.split(',')]
    config.TRAIN.GPU_NUM = len(gpus)


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
        config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
        config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
        config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    data = cfg.DATASET.NAME
    full_name = '{name}_{data}'.format(name=name, data=data)

    return full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
