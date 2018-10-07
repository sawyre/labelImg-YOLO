#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import csv
from utils import parse_annotation_xml
from compute_metrics import get_numbers

# Папки, в которых лежат истинные и предсказанные аннотации
anno_gt = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/2-TwoPopular/val/annotations/'
anno_pred = '/media/data/ObjectDetectionExperiments/Projects/2_YOLOs/YOLOv2_Orlova/Experiencor/Results/20/annotations/'
# Минимально допустимое значение IoU для сравниваемых боксов
min_overlap = 0.3
# Целевые классы
classes = ('5.19.1', '2.1')
# Значения минимально допустимой уверенности, для которых ведем расчет
min_conf = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
min_conf = 0.3
# Файл .csv, куда сохраним промежуточные значения по файлам (для рабора вручную)
csv_output = '/media/data/ObjectDetectionExperiments/Projects/2_YOLOs/YOLOv2_Orlova/Experiencor/Results/20/stat1.csv'

nb_true       = 0
nb_falseneg   = 0
nb_falseclass = 0
nb_falsebox   = 0

gt     = set(os.listdir(anno_gt))
pred   = set(os.listdir(anno_pred))

match = gt.intersection(set(pred))
l_match = len(match)

print('Найдено {} совпадающих пар аннотаций'.format(l_match))

if csv_output:
    with open(csv_output, 'w') as file:
        # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
        wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
        wr.writerow(['##############\tminIoU\t{:1.2f}\tminConf\t{:1.2f}'.format(min_overlap, min_conf)])
        wr.writerow(['Файл аннотации\tn_true\tn_falseneg\tn_falsepos\tl_gtboxes\tl_predboxes'])

numbers = np.zeros(shape=(len(classes), 3), dtype=int)
for i, m in enumerate(match):
    gt_imgname,     gt_img_w,   gt_img_h,   gt_img_d,   gt_boxes = parse_annotation_xml(anno_gt + m)
    pred_imgname, pred_img_w, pred_img_h, pred_img_d, pred_boxes = parse_annotation_xml(anno_pred + m)
    #  Сверим, что название изображения и его размеры одинаковы
    '''
    assert (gt_imgname==pred_imgname and
            gt_img_w==pred_img_w     and
            gt_img_h==pred_img_h     and
            gt_img_d==pred_img_d), "В аннотациях {} различаются имя изображения или его размеры!".format(m)
            '''
    assert (gt_imgname == pred_imgname), "В аннотациях {} различаются имя изображения или его размеры!".format(m)

    # *Отфильтруем предсказанные боксы со слишком низкой достоверностью
    pred_boxes = list(filter(lambda x: x[0] < min_conf, pred_boxes))

    # Теперь посчитаем метрики
    img_numbers = get_numbers(gt_boxes   = gt_boxes,
                              pred_boxes = pred_boxes,
                              classes    = classes,
                              min_conf   = min_conf,
                              min_IoU    = min_overlap)
    l_gt = len(gt_boxes)
    l_pred = len(pred_boxes)
    true = np.sum(img_numbers[:, 0])
    fneg = np.sum(img_numbers[:, 1])
    fpos = np.sum(img_numbers[:, 2])

    numbers = np.add(numbers, img_numbers)

    if csv_output:
        with open(csv_output, 'a') as file:
            # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
            wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
            wr.writerow(['{}\t{:2d}\t{:2d}\t{:2d}\t{:2d}\t{:2d}'.format(m, true, fneg, fpos, l_gt, l_pred)])

    print('Изображение {} обработано\t\t'
          'gtboxes {:2d}, prboxes {:2d}, true {:2d}, fneg {:2d}, fpos {:2d}'.format(m, l_gt, l_pred, true, fneg, fpos))
    print('')


