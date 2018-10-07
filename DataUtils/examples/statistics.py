#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import csv
import math
from stat_utils import parse_annotation_xml
from compute_metrics import get_numbers, get_confMatrix

from map_lib import calc_recall_precision
from map_lib import calc_mean_ap
from map_lib import calc_auc

class Stat:

    def __init__(self, anno_gt, anno_pred, classes):
        self.classes = classes
        self.anno_gt = anno_gt
        self.anno_pred = anno_pred

        gt = set(os.listdir(anno_gt))
        pred = set(os.listdir(anno_pred))

        self.match = gt.intersection(set(pred))
        self.l_match = len(self.match)

        print('Найдено {} совпадающих пар аннотаций'.format(self.l_match))


    def AP(self, min_overlap, min_conf, csv_output='', strict_true = True, raw_nums = False):
    # Здесь мы считаем среднюю точность для заданного набора minIoU, minConfidence по всем изображениям

        if csv_output:
            with open(csv_output, 'a') as file:
                # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
                wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
                wr.writerow(['##############\tminIoU\t{:1.2f}\tminConf\t{:1.2f}'.format(min_overlap, min_conf)])
                wr.writerow(['Файл аннотации\tn_true\tn_falseneg\tn_falsepos'])

        numbers = np.zeros(shape=(len(self.classes), 3), dtype=int)
        
        yugreks = [[],[]]
        yugreks_class = {}

        for cls in classes:
               yugreks_class[cls]=[[],[]]   

        for i, m in enumerate(self.match):
            gt_imgname,     gt_img_w,   gt_img_h,   gt_img_d,   gt_boxes = parse_annotation_xml(self.anno_gt + m)
            pred_imgname, pred_img_w, pred_img_h, pred_img_d, pred_boxes = parse_annotation_xml(self.anno_pred + m)
            #  Сверим, что название изображения и его размеры одинаковы
            '''
            assert (gt_imgname==pred_imgname and
                    gt_img_w==pred_img_w     and
                    gt_img_h==pred_img_h     and
                    gt_img_d==pred_img_d), "В аннотациях {} различаются имя изображения или его размеры!".format(m)
                    '''
            assert (gt_imgname == pred_imgname), "В аннотациях {} различаются имя изображения или его размеры!".format(m)

            # *Отфильтруем предсказанные боксы по порогу
            # В список попадают лишь те элементы, для которых условие верно (ф-я выдает True)
            pred_boxes = list(filter(lambda x: not(x[0] < min_conf), pred_boxes))


            # Теперь посчитаем метрики
            img_numbers, img_yugreks, img_yugreks_class = get_numbers(gt_boxes    = gt_boxes,
                                      pred_boxes  = pred_boxes,
                                      classes     = self.classes,
                                      min_conf    = min_conf,
                                      min_IoU     = min_overlap,
                                      strict_true = strict_true)
            l_gt = len(gt_boxes)
            l_pred = len(pred_boxes)
            true = np.sum(img_numbers[:, 0])
            fneg = np.sum(img_numbers[:, 1])
            fpos = np.sum(img_numbers[:, 2])

            yugreks[0]+=img_yugreks[0]
            yugreks[1]+=img_yugreks[1]
            for cls in classes:
                yugreks_class[cls][0]+=img_yugreks_class[cls][0]
                yugreks_class[cls][1]+=img_yugreks_class[cls][1]

            numbers = np.add(numbers, img_numbers)

            if csv_output:
                with open(csv_output, 'a') as file:
                    # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
                    wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
                    wr.writerow(['{}\t{:2d}\t{:2d}\t{:2d}'.format(m, true, fneg, fpos)])

            if i%100:
                print('Изображение {}/{} {} обработано\t\t'
                      'gtboxes {:2d}, prboxes {:2d}, true {:2d}, fneg {:2d}, fpos {:2d}'.format(str(i+1), str(self.l_match), m, l_gt, l_pred, true, fneg, fpos))

        if csv_output:
            with open(csv_output, 'a') as file:
                # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
                wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
                for i, c in enumerate(self.classes):
                    wr.writerow(['\n'])
                    wr.writerow(['{}\t{:2d}\t{:2d}\t{:2d}'.format(c, numbers[i, 0], numbers[i, 1], numbers[i, 2])])

        precision = numbers[:, 0].astype(float) / (numbers[:, 0] + numbers[:, 2]).astype(float)
        recall    = numbers[:, 0].astype(float) / (numbers[:, 0] + numbers[:, 1]).astype(float)

        #prc_all = {}
        auc_map_list = []
        for cls in classes:
            

            key_scores = yugreks_class[cls][0]
            key_Ys = yugreks_class[cls][1]
            key_prc = calc_recall_precision(key_scores, key_Ys)
            key_prec = [x[1] for x in key_prc]
            key_rec = [x[0] for x in key_prc]

            #prc_all[key] = [key_prec, key_rec, key_prc]
            
            auc = calc_auc(key_prc)
            mean_ap = calc_mean_ap(key_prc)
            
            auc_map_list.append([auc, mean_ap, cls])

        all_scores = yugreks[0]
        all_Ys = yugreks[1]
        all_prc = calc_recall_precision(all_scores, all_Ys)
        all_auc = calc_auc(all_prc)
        all_mean_ap = calc_mean_ap(all_prc)

        all_map = [all_mean_ap, all_auc]
            
            
        for p in precision:
            if math.isnan(p):
                p = 0.0
        for r in recall:
            if math.isnan(r):
                r = 0.0

        if raw_nums:
            return precision, recall, numbers[:, 0], numbers[:, 1], numbers[:, 2], auc_map_list, all_map
        else:
            return precision, recall, auc_map_list, all_map

    def AP_dir(self, min_overlaps, min_confidences, csv_output, csv_output_ext='', strict_true = True, raw_nums = False):

        with open(csv_output, 'w') as file:
            # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
            wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
            wr.writerow(['\n'])

        if csv_output_ext:
            with open(csv_output_ext, 'w') as file:
                # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
                wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
                wr.writerow(['\n'])

        linelist = []
        for min_overlap in min_overlaps:

            hat = '#########\t'
            line = 'minConf\t'
            for i, c in enumerate(classes):
                
                if raw_nums:
                    hat += '|######\t#########\t#########\tclass\t{}\t#########\t#########\t########|\t'.format(classes[i])
                    line += 'precision\trecall\tp*r\ttrue\tfalseneg\tfalsepos\tmAP\tAUC\t'
                else:
                    hat += '|######\tclass\t{}\t#########\t########|\t'.format(classes[i])
                    line += 'precision\trecall\tp*r\tmAP\tAUC\t'

            hat+='|######\tall\t########|\t'
            line += '|\tmAP\tAUC\t'

            linelist.append('minIoU\t{:1.2f}'.format(min_overlap))
            linelist.append(hat)
            linelist.append(line)

            for min_conf in min_confidences:
                result = self.AP(min_overlap=min_overlap,
                                       min_conf=min_conf,
                                       csv_output=csv_output_ext,
                                       strict_true=strict_true,
                                       raw_nums=raw_nums)

                line = '{:1.2f}\t'.format(min_conf)

                if raw_nums:
                    precisions, recalls, trues, fnegs, fposs, auc_map, all_map = result
                    for i, c in enumerate(classes):
                        line += '{:1.4f}\t{:1.4f}\t{:1.4f}\t{:d}\t{:d}\t{:d}\t{:1.4f}\t{:1.4f}\t'.format(precisions[i],
                                                                                       recalls[i],
                                                                                       precisions[i] * recalls[i],
                                                                                       trues[i],
                                                                                       fnegs[i],
                                                                                       fposs[i],
                                                                                       auc_map[i][1],
                                                                                       auc_map[i][0])

                    line += '|\t{:1.4f}\t{:1.4f}\t'.format(all_map[0], all_map[1]);
                    

                else:
                    precisions, recalls, auc_map, all_map = result
                    for i, c in enumerate(classes):
                        line += '{:1.4f}\t{:1.4f}\t{:1.4f}\t{:1.4f}\t{:1.4f}\t'.format(precisions[i], recalls[i],
                                                                         precisions[i] * recalls[i],
                                                                           auc_map[i][1],
                                                                           auc_map[i][0])
                    line += '|\t{:1.4f}\t{:1.4f}\t'.format(all_map[0], all_map[1]);

                linelist.append(line)

        with open(csv_output, 'a') as file:
            # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
            wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
            wr.writerow(linelist)

        print('Cтатистика сохранена в файл {},\nболее подробная информация  - в файл {}'.format(csv_output, csv_output_ext))

    def confusionMatrix(self, min_overlaps, min_confidences, csv_output='', strict_true = False):

        with open(csv_output, 'w') as file:
            # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
            wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
            wr.writerow(['\n'])

        matclasses = list(self.classes)
        matclasses.append('none')

        linelist = []

        for i_over, min_overlap in enumerate(min_overlaps):

            for i_conf, min_conf in enumerate(min_confidences):

                matrix = np.zeros(shape=(len(matclasses), len(matclasses)), dtype=int)

                linelist.append(' ')
                linelist.append('minIoU\t{}\nminConf\t{}'.format(min_overlap, min_conf))
                hat = 'pred\\true\t'
                for i, c in enumerate(matclasses):
                    hat += '{}\t'.format(matclasses[i])
                    print('')
                linelist.append(hat)

                for i, m in enumerate(self.match):
                    gt_imgname,     gt_img_w,   gt_img_h,   gt_img_d,   gt_boxes = parse_annotation_xml(self.anno_gt + m)
                    pred_imgname, pred_img_w, pred_img_h, pred_img_d, pred_boxes = parse_annotation_xml(self.anno_pred + m)
                    #  Сверим, что название изображения и его размеры одинаковы
                    '''
                    assert (gt_imgname==pred_imgname and
                            gt_img_w==pred_img_w     and
                            gt_img_h==pred_img_h     and
                            gt_img_d==pred_img_d), "В аннотациях {} различаются имя изображения или его размеры!".format(m)
                            '''
                    assert (gt_imgname == pred_imgname), "В аннотациях {} различаются имя изображения или его размеры!".format(m)

                    # *Отфильтруем предсказанные боксы со слишком низкой достоверностью
                    # В список попадают лишь те элементы, для которых условие верно (ф-я выдает True)
                    pred_boxes_filtered = list(filter(lambda x: not(x[0] < min_conf), pred_boxes))

                    # Теперь посчитаем матрицу
                    img_matrix = get_confMatrix(gt_boxes   = gt_boxes,
                                                pred_boxes = pred_boxes_filtered,
                                                classes    = matclasses,
                                                min_conf   = min_conf,
                                                min_IoU    = min_overlap)

                    matrix += img_matrix
                    print('IoU={}, Conf={}, изображение {}/{} обработано.'.format(min_overlap, min_conf, i+1, self.l_match))


                # Запишем её в выходную строку
                for i_c, c in enumerate(matclasses):
                    line = '{}\t'.format(c)
                    row = matrix[i_c, :]
                    for i_item, item in enumerate(row):
                        line += '{}\t'.format(item)
                    linelist.append(line)

                print('Часть {}/{} обработана. Результат будет в файле {}'.format(len(min_confidences)*i_over+i_conf+1, len(min_overlaps)*len(min_confidences), csv_output))

        # Запишем выходную строку
        with open(csv_output, 'a') as file:
            # escape char - для разделения на столбцы, delimiter - для перехода на новую строку
            wr = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\t', delimiter='\n')
            wr.writerow(linelist)

        print('Confusion matrix have been done!')


log = 22
dataset = ''
d = ''
# Целевые классы
classes = ("MarkerTC_Circle",
"DU_Right_Hole",
"Man_Begin",
"Under_Cross_Dbl_Part",
"Right_Corner",
"MarkerBC_Circle",
"MarkerLC_Circle",
"DUnit_Center_Small",
"Under_DUnit_Box",
"DUnit_LeftAnt_n2",
"DUnit_LeftAnt_n3",
"Triple_Part_Left",
"Circle_Top_Hole",
"Right_Edge_Part",
"Comb_Cell_Part",
"MarkerBC_Rhombus",
"Left_Rect",
"TU_DoublePart",
"DUnit_Part_3",
"DUnit_Part_2",
"DUnit_Part_1",
"MarkerC_Rhombus",
"DUnit_Part_5",
"DUnit_Part_4",
"MarkerTC_Rhombus",
"Marker_Ant",
"DUnit_Circle_Top",
"Module_Edge",
"Module_Slot",
"Top_Corner",
"Right_Circle_Part",
"Left_Ant",
"DUnit_Center",
"Under_Marker_Ledge",
"Comb_Double_Part_n4",
"Comb_Double_Part_n3",
"Comb_Double_Part_n2",
"SP_Left",
"DUnit_Left_Marker",
"TopUnit",
"DUnit",
"Ledge_n4",
"Right_Unit",
"Ledge_n2",
"Ledge_n3",
"DUnit_Ledge",
"DU_Right_Bm_Part",
"DUnit_Left_Hole",
"Double_Ant",
"Left_Corner",
"Cross_Right",
"Left_Unit",
"DU_Right_Top_Part",
"Bottom_Double_Part",
"DUnit_BottomAnt",
"Circle_Bottom_Hole",
"SP_Left_n4",
"Marker_Rhombus",
"Right_Cell_Part",
"Right_Part",
"DUnit_LeftCircleAnt_n2",
"Left_Double_Part",
"Left_Edge_Part",
"Cross_Left",
"Black_Rect",
"Triple_Part_Center",
"MarkerC_Circle",
"Right_Double_Part",
"DUnit_RightAnt_n2",
"DUnit_RightAnt_n3",
"Left_Part",
"Left_Cell_Part",
"Right_Rect",
"Double_Part",
"Double_Part_2",
"DUnit_Right_Hole",
"DU_Right_Dbl_Part",
"Cross",
"DUnit_Circle_Bottom",
"MarkerRC_Rhombus",
"Triple_Part_Right",
"DUnit_RightCircleAnt_n2",
"MarkerLC_Rhombus",
"MarkerRC_Circle",
"SP_Right",
"DUnit_Triple_Part",
"Circle_Part",
"Top_Double_Part",
"Rail_Edge",
"Marker_Circle",
"Black_Rect_Top",
"SP_Right_n4",
"DUnit_RightCircleAnt_n3",
"TU_Circle",
"DUnit_Top",
"Rect_Block") #Список классов сюда?

#Для тестирования параметров изображения:
#dirs = [['Images.Normal', 'yolo-normal'],['Images.Contrast', 'yolo-contrast'], ['Images.eh.contr', 'yolo-eh-contrast'], ['Images.EqHist', 'yolo-eq-hist'], ['Images.Contr.Rez', 'yolo-contr-rez']]

#Для тестирования разных узлов
node_path = '/home/ivan/database/test_by_node/'
dirs = [['Node2', 'Node2/yolo-normal']]#,['Node3', 'Node3/yolo-normal'],['Node4', 'Node4/yolo-normal']]

# Папки, в которых лежат истинные и предсказанные аннотации

for pair in dirs:
    basic_path = node_path+pair[0]+'/'
    data_dir = pair[1]
    anno_gt = basic_path+'output/VOC_annotations/'
    anno_pred = node_path+'/'+data_dir+'/annotations/'
    stat_dir = node_path+'/'+data_dir+'/stats'
    if os.path.exists(stat_dir):
        shutil.rmtree(stat_dir, ignore_errors = True)

    os.makedirs(stat_dir)
    # Минимально допустимое значение IoU для сравниваемых боксов
    min_overlaps = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    # Значения минимально допустимой уверенности, для которых ведем расчет
    min_confidences = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    # Файл .csv, куда сохраним промежуточные значения по файлам (для рабора вручную)
    csv_output_ap = stat_dir+'/stat_AP.csv'
    csv_output_ext = stat_dir+'/stat_ext.csv'
    csv_output_mat = stat_dir+'/stat_matrix.csv'
    # Если True, то считаем, что каждый предсказанный бокс может совпадать лишь с одним истинным боксом и наоборот.
    strict_true = False
    # Если True. то выведем в статистику и абсолютные кол-ва верных и ложных предсказаний
    raw_nums = True

    statistics = Stat(anno_gt=anno_gt, anno_pred=anno_pred, classes=classes)

    statistics.AP_dir(min_overlaps = min_overlaps,
                      min_confidences = min_confidences,
                      csv_output=csv_output_ap,
                      csv_output_ext=csv_output_ext,
                      strict_true=strict_true,
                      raw_nums=raw_nums)

    '''
    statistics.confusionMatrix(min_overlaps = min_overlaps,
                               min_confidences = min_confidences,
                               csv_output=csv_output_mat,
                               strict_true=strict_true)
    '''

