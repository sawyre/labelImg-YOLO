#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script takes in a configuration file and produces the best model.
The configuration file is a json file and looks like this:

{
    "model" : {
        "architecture":         "Full Yolo",
        "input_size":           416,
        "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "max_box_per_image":    10,
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/andy/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/andy/data/raccoon_dataset/anns/",

        "train_times":          10,
        "pretrained_weights":   "",
        "batch_size":           16,
        "learning_rate":        1e-4,
        "nb_epoch":             50,
        "warmup_batches":       100,

        "object_scale":         5.0 ,
        "no_object_scale":      1.0,
        "coord_scale":          1.0,
        "class_scale":          1.0,

        "debug":                true
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",

        "valid_times":          1
    }
}
"""

import os
import numpy as np
from preprocessing import parse_annotation
from frontend import YOLO
import json
import tensorflow as tf
import keras.backend as K

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def trainer(config_path, logdir = '~/logs/'):
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################

    # parse annotations of the training set
    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    # parse annotations of the validation set, if any, otherwise split the training set
    if os.path.exists(config['valid']['valid_annot_folder']):
        valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                    config['valid']['valid_image_folder'],
                                                    config['model']['labels'])
    else:
        train_valid_split = int(0.8 * len(train_imgs))
        np.random.shuffle(train_imgs)
        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()



    ###############################
    #   Construct the model
    ###############################

    yolo = YOLO(architecture      = config['model']['architecture'],
                input_size        = config['model']['input_size'],
                labels            = config['model']['labels'],
                max_box_per_image = config['model']['max_box_per_image'],
                anchors           = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any)
    ###############################

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in {}".format(config['train']['pretrained_weights']))

        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process
    ###############################

    yolo.train(train_imgs         = train_imgs,
               valid_imgs         = valid_imgs,
               train_times        = config['train']['train_times'],
               valid_times        = config['valid']['valid_times'],
               nb_epoch           = config['train']['nb_epoch'],
               learning_rate      = config['train']['learning_rate'],
               batch_size         = config['train']['batch_size'],
               warmup_epochs      = config['train']['warmup_epochs'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               saved_weights_name = config['train']['saved_weights_name'],
               debug              = config['train']['debug'],
               logdir             = logdir)

# Пример использования
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.allocator_type = 'BFC'
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.85

logs = 'logs/nissan2/'
config = 'config_for_train.json'
import shutil
shutil.copy(config, logs+'config.json')

trainer(config_path = config, logdir = logs)
