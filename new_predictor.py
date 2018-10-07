#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import get_annoboxes, draw_boxes
from frontend import YOLO
import json
from timeit import default_timer as timer
from PIL import Image
from convert_to_xml import save_anno_xml
import shutil
import threading

class predictor:

    def __init__(self, config_path, weights_path):
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        self.labels = config['model']['labels']

        self.yolo = YOLO(architecture     = config['model']['architecture'],
                        input_size        = config['model']['input_size'],
                        labels            = self.labels,
                        max_box_per_image = config['model']['max_box_per_image'],
                        anchors           = config['model']['anchors'])

        self.yolo.load_weights(weights_path)

    def _predict_one(self, image, threshold, decimals, draw_bboxes=True):
        boxes = self.yolo.predict(image, threshold=threshold)
        #image = draw_boxes(image, boxes, self.labels, decimals=decimals)
        print(str(len(boxes)) + 'boxes are found')
        return image, boxes

    def predict_from_dir(self, qtclass, path_to_dir, path_to_outputs = None, threshold=0.5, decimals=8, save_anno=False, draw_truth=False):
        if path_to_outputs and not os.path.exists(path_to_outputs):
            print('Creating output path {}'.format(path_to_outputs))
            os.mkdir(path_to_outputs)

        count_of_images = len([name for name in os.listdir(path_to_dir)])
        number_labeled_images = 0
        for image_filename in os.listdir(path_to_dir):
            if (qtclass.stop_progress):
                break
            # TODO: здесь надо сделать адекватную проверку, изображение ли это
            if image_filename.endswith('bmp') or image_filename.endswith('jpg') or image_filename.endswith('png') or image_filename.endswith('jpeg'):
                image = cv2.imread(path_to_dir + image_filename, cv2.IMREAD_COLOR)
                image_h = image.shape[0]
                image_w = image.shape[1]

                curr_time = timer()

                image, boxes = self._predict_one(image, threshold=threshold, decimals=decimals)
                number_labeled_images += 1

                curr_time = timer() - curr_time
                print(curr_time)

                boxes = get_annoboxes(image_w=image_w, image_h=image_h, boxes = boxes, labels=self.labels)

                if path_to_outputs:

                    if save_anno:
                        #
                        if (image_filename.endswith('jpeg')):
                            save_anno_xml(dir=path_to_outputs,
                                          img_name=image_filename[:-5],
                                          img_format=image_filename[-4:],
                                          img_w=image.shape[1],
                                          img_h=image.shape[0],
                                          img_d=image.shape[2],
                                          boxes=boxes,
                                          quiet=False,
                                          minConf=threshold)
                        else:
                            save_anno_xml(dir=path_to_outputs,
                                          img_name=image_filename[:-4],
                                          img_format=image_filename[-3:],
                                          img_w=image.shape[1],
                                          img_h=image.shape[0],
                                          img_d=image.shape[2],
                                          boxes=boxes,
                                          quiet=False,
                                          minConf=threshold)

                    #retval = cv2.imwrite(path_to_outputs + 'images/' + image_filename, image)
                    #if retval:
                    #    print('Изображение {} успешно сохранено в папку {}'.format(image_filename, path_to_outputs))
            else:
                print('В папке не только изображения - {}'.format(image_filename))
            if(count_of_images != 0):
                qtclass.setProgress(number_labeled_images * 100 / count_of_images)
            else:
                qtclass.cancel()


#'''
def labeling(img_for_pred, img_results, config, weights, thres, qtclass):
    #logdir  = 22

    pred = predictor(config_path=config,
                     weights_path=weights)


    if os.path.exists(img_for_pred):
            #continue
        img_for_pred+='/'
        #img_results  = node_path+pair[1]
        if os.path.exists(img_results):
            shutil.rmtree(img_results, ignore_errors = True)

        os.makedirs(img_results)
        img_results+='/'
        #os.makedirs(img_results+'annotations')
        #os.makedirs(img_results+'images')

        pred.predict_from_dir(qtclass=qtclass,
                              path_to_dir=img_for_pred,
                              path_to_outputs=img_results,
                              threshold=thres,
                              save_anno=True,
                              decimals=8)
#'''
