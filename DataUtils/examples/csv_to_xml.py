#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import shutil
from xml.etree.ElementTree import Element, ElementTree, SubElement
import csv
from utils import getFormat, parse_annotation_xml, get_roi_image, get_roi_anno, save_anno_xml


# пример с RTSD
imgdir     = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/ORIG/images/'
anno       = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/full-gt.csv'
output_ann = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/ORIG/annotations/'
#output_img = '/media/data/ObjectDetectionExperiments/Datasets/43_LISA/images/'
img_suffix = ''

# Получим данные из аннотаций
with open(anno) as csv_file:
    reader = csv.reader(csv_file)
    anno = [row for row in reader]
# Удалим все пустые строки
anno = list(filter(None, anno))

# составим список изображений из папки с ними
imgnames = os.listdir(imgdir)
# Определим формат изображений
img_suffix, pos_suffix = getFormat(imgnames[0])

n = len(imgnames)

for i, imgname in enumerate(imgnames):
    bboxes = []
    # Получим боксы для этого изображения
    for a in anno[1:]:
        if a[0] == imgname:
            a[3] = str(int(a[1])+int(a[3]))
            a[4] = str(int(a[2])+int(a[4]))
            #Эта часть для единой полной базы.
            a[5] = a[5].replace('_', '.')
            #a[6] - это id для трекера
            box = (a[6], a[5], a[1], a[2], a[3], a[4])
            bboxes.append(box)
    # Выясним размеры изображения
    img = cv2.imread(imgdir+imgname)
    if len(img.shape) > 2:
        h, w, d = img.shape
    else:
        h, w = img.shape

    # Запишем данные в xml-файл,img_name БЕЗ .xml
    save_anno_xml(dir=output_ann,
              img_name=imgname[:pos_suffix-1],
              img_format=img_suffix,
              img_w=w,
              img_h=h,
              img_d=d,
              boxes=bboxes,
              quiet=False,
              minConf=-1.)
    print('Сохранили аннотацию {}/{}'.format(i+1, n))


