#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from stat_utils import intersection
import os

def IoU(box1, box2):
    # Найдем площади данных боксов
    s1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
    s2 = (box2[3] - box2[1]) * (box2[2] - box2[0])
    # Найдем точки бокса-пересечения

    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    dx = x2 - x1
    dy = y2 - y1
    if dx > 0 and dy > 0:
        # Найдем площадь бокса-пересечения
        i = dx * dy
        u = s1 + s2 - i
        iou = np.round(float(i)/u, 3)
        return iou
        #
    else:
        return 0.

def get_numbers(gt_boxes, pred_boxes, classes, min_conf, min_IoU=0.3, strict_true = True):

    numbers = np.zeros(shape=(len(classes), 3), dtype=int)
    
    yugreks_class = {}
    yugreks = [[], []]

    for classname in classes:
        yugreks_class[classname]=[[], []]
    
    # Получим матрицу перекрытий боксов по классам
    for i_class, classname in enumerate(classes):
        true = 0        # верные обнаружения
        falseneg = 0    # ложноотрицательные (пропущенные объекты)
        falsepos = 0    # ложноположительные (несуществующие объекты)
        pred_boxes_class = list(filter(lambda x: x[-5] == classname, pred_boxes))
        gt_boxes_class = list(filter(lambda x: x[-5] == classname, gt_boxes))
        #overlap_matrix = np.zeros(shape=(len(gt_boxes_class), len(pred_boxes_class)), dtype=float)

        if len(pred_boxes_class) > 0:
            pred_boxes_class.sort(key=lambda x: x[0], reverse=True)
            gt_boxes_class2 = list(gt_boxes_class)
            for i_pred, box_pred in enumerate(pred_boxes_class):
                # Если в списке истинных боксов что-то осталось:
                y=0
                conf=pred_boxes_class[0]
                if len(gt_boxes_class2) > 0:
                    ious = []
                    [ious.append((IoU(box_gt[-4:], box_pred[-4:]), i_gt)) for i_gt, box_gt in enumerate(gt_boxes_class2)]
                    ious.sort(key = lambda x: x[1], reverse=True)
                    # Если может быть только одно совпадение для каждого бокса
                    if strict_true:
                        iou = ious[0]
                        # Если один из истинных боксов перекрывается в достаточной степени с предсказанным боксом
                        if iou[0] >= min_IoU:
                            # Увеличим счетчик верных обнаружений на 1
                            true += 1
                            y=1
                            # Удалим использованный истинный бокс
                            del gt_boxes_class2[iou[1]]
                    # Если может быть много совпадений для каждого бокса
                    else:
                        for iou in ious:
                            if iou[0] >= min_IoU:
                                true += 1
                                y=1
                yugreks[0].append(conf)
                yugreks[1].append(y)
                yugreks_class[classname][0].append(conf)
                yugreks_class[classname][1].append(y)

        # Остальные числа можно определить, зная число верных предсказаний и число истинных и предсказанныx боксов
        falseneg = max(len(gt_boxes_class) - true, 0)
        falsepos = max(len(pred_boxes_class) - true, 0)
        numbers[i_class, 0] = true
        numbers[i_class, 1] = falseneg
        numbers[i_class, 2] = falsepos

    return numbers, yugreks, yugreks_class

def get_confMatrix(gt_boxes, pred_boxes, classes, min_conf, min_IoU=0.3, strict_true = False):

    matrix = np.zeros(shape=(len(classes), len(classes)), dtype=int)
    if len(pred_boxes) == 0:
        if len(gt_boxes) > 0:
            for gt_box in gt_boxes:
                matrix[-1, classes.index(gt_box[-5])] += 1
        return matrix

    elif len(gt_boxes) == 0:
        for pred_box in gt_boxes:
            matrix[classes.index(pred_box[-5], -1)] += 1
        return matrix

    gt_boxes.sort(key = lambda x: x[0], reverse = True)
    pred_boxes.sort(key = lambda x: x[0], reverse=True)
    gt_boxes2   = list(gt_boxes)
    gt_boxes_used = [False for box in gt_boxes2]

    # Сначала найдем совпадения
    for i_pred, box_pred in enumerate(pred_boxes):
        box_pred_isUsed = False
        # Если в списке истинных боксов что-то осталось:
        if len(gt_boxes2) > 0:
            ious = []
            [ious.append((IoU(box_gt[-4:], box_pred[-4:]), i_gt)) for i_gt, box_gt in enumerate(gt_boxes2)]
            ious.sort(key=lambda x: x[1], reverse=True)

            # Если может быть только одно совпадение для каждого бокса
            if strict_true:
                iou = ious[0]
                # Если один из истинных боксов перекрывается в достаточной степени с предсказанным боксом
                if iou[0] >= min_IoU:
                    # Добавим единицу в нужную ячейку матрицы
                    box_gt = gt_boxes2[iou[1]]
                    c_gt   = box_gt[-5]
                    c_pred = box_pred[-5]
                    matrix[classes.index(c_pred),classes.index(c_gt)] += 1
                    box_pred_isUsed = True
                    # Удалим использованный истинный бокс
                    del gt_boxes2[iou[1]]

            # Если может быть много совпадений для каждого бокса
            else:
                for iou in ious:
                    if iou[0] >= min_IoU:
                        # Добавим единицу в нужную ячейку матрицы
                        box_gt = gt_boxes2[iou[1]]
                        c_gt = box_gt[-5]
                        c_pred = box_pred[-5]
                        matrix[classes.index(c_pred), classes.index(c_gt)] += 1
                        # Отметим использованные боксы
                        box_pred_isUsed = True
                        gt_boxes_used[iou[1]] = True

        if not box_pred_isUsed:
            matrix[classes.index(box_pred[-5]), -1] += 1

    # Рассуём оставшиеся значения истинных боксов
    if strict_true:
        for gt_box in gt_boxes2:
            matrix[-1, classes.index(gt_box[-5])] += 1
    else:
        for gt_box, isUsed in zip(gt_boxes2, gt_boxes_used):
            if not isUsed:
                matrix[-1, classes.index(gt_box[-5])] += 1

    return matrix




