#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import random
from utils import getFormat, parse_annotation_xml, get_roi_image, get_roi_anno, save_anno_xml
import sys

input_images   = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/ORIG/images/'
input_annos    = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/ORIG/annotations/'
output_cropped_train = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/4-SuperSign/train/images/'
output_annos_train   = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/4-SuperSign/train/annotations/'
output_cropped_val   = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/4-SuperSign/val/images/'
output_annos_val     = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/4-SuperSign/val/annotations/'
images_list    = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/all.list'

classes        = ()  # если не поставим запятую c одним классом, будет строка, а не tuple
ignor          = ()                 # объектов этих классов не будет в аннотациях
default_name   = 'sign'             # все обнаруженные классы, не принадлежащие classes и ignor,
                                    # будут переименованы этим именем
numroi         = 5      # сколько ОИ делаем из каждого изображения (в дело из них пойдут лишь те, на которых
                        # есть искомые объекты из classes)
numroi_val     = 3
resize         = True   # Будем ли слегка менять пропорции изображения (пока не работает)
val_split      = 0.07   # Если не 0, то часть данных будет для валидации

# Парсим список целевых изображений
with open(images_list, 'r') as f:
    file = f.read()
targets = file.split(sep='\n')
# ДВА последних элемента - пустая строка, по который мы ранее поняли, что файл закончился
targets = list(filter(None, targets))
targets = set(targets)
n = len(targets)
print('Всего берем {} изображений'.format(n))

if val_split > 0:
    n_val = int(n*val_split)
    if n_val > 0:
        if n_val > 3000:
            n_val = 3000
        targets_val = set(random.sample(targets, n_val))
        targets = targets - targets_val
        n = len(targets)
        print('На обучение идет {}, а на валидацию {} изображений'.format(n, n_val))
        targets = list(targets)
        targets_val = list(targets_val)

        for j,target in enumerate(targets_val):
            # Определим формат изображения и индекс точки, отделяющей расширение
            format, k = getFormat(target)
            # Определим имя файла без расширения И ТОЧКИ, чтобы найти аннотацию c таким же именем
            base_name = target[:k-1]
            print('Работаем с изображением {}'.format(target))
            # Загрузим и распарсим аннотацию
            imgname, img_w, img_h, img_d, boxes = parse_annotation_xml(input_annos+base_name+'.xml')
            assert imgname == target, 'VAL\tЧто-то не то с аннотациями, имя изображения {} в аннотации не совпадает!'.format(target)
            if len(boxes) == 0:
                print('VAL\tТут нет боксов!')
                continue
            # Получим заданное кол-во ОИ
            res, rois = get_roi_image(image_w=img_w, image_h=img_h, target_w=608, target_h=608, number=numroi_val)
            assert res, 'VAL\tНе смогли получить ОИ!'
            # Теперь получим новые координаты для каждой ОИ
            # и сохраним всю эту прелесть в новое изображение и аннотацию.
            for i, roi in enumerate(rois):
                new_boxes = get_roi_anno(roi_box=roi,
                                         obj_boxes=boxes,
                                         target_classes=classes,
                                         ignored_classes=ignor,
                                         rename_as=default_name,
                                         min_overlap=0.6)
                if len(new_boxes)==0:
                    print('VAL\tВ этой roi {}/{} нет нужных нам объектов'.format(i+1, len(rois)))
                    # Значит, в этой roi нет нужных нам объектов
                    continue

                # Если мы здесь, значит у нас есть нужные нам объекты и у нас уже даже есть их новые координаты!
                print('VAL\tВ этой roi {}/{} есть нужные нам объекты'.format(i + 1, len(rois)))
                    # Cначала вырежем и сохраним нашу ОИ
                name = base_name + '_{:02d}_{:02d}'.format(j+1,i+1) # имя без точки и расширения!
                image = cv2.imread(input_images + target)
                if img_d==3:
                    image = image[roi[1]:roi[3], roi[0]:roi[2], :]
                if img_d==1:
                    image = image[roi[1]:roi[3], roi[0]:roi[2]]
                cv2.imwrite(output_cropped_val + name+'.'+format, image)
                    # Теперь сохраним аннотацию
                save_anno_xml(dir = output_annos_val,
                              img_name = name,
                              img_format=format,
                              img_w=roi[2]-roi[0],
                              img_h=roi[3]-roi[1],
                              img_d=img_d,
                              boxes=new_boxes,
                              quiet=True)

            print('VAL\tСохранили новые изображения и аннотации {}/{}'.format(j+1, n_val))


for j,target in enumerate(targets):
    # Определим формат изображения и индекс точки, отделяющей расширение
    format, k = getFormat(target)
    # Определим имя файла без расширения И ТОЧКИ, чтобы найти аннотацию c таким же именем
    base_name = target[:k-1]
    print('Работаем с изображением {}'.format(target))
    # Загрузим и распарсим аннотацию
    imgname, img_w, img_h, img_d, boxes = parse_annotation_xml(input_annos+base_name+'.xml')
    assert imgname == target, 'TRAIN\tЧто-то не то с аннотациями, имя изображения {} в аннотации не совпадает!'.format(target)
    if len(boxes) == 0:
        print('TRAIN\tТут нет боксов!')
        continue
    # Получим заданное кол-во ОИ
    res, rois = get_roi_image(image_w=img_w, image_h=img_h, target_w=608, target_h=608, number=numroi)
    assert res, 'TRAIN\tНе смогли получить ОИ!'
    # Теперь получим новые координаты для каждой ОИ
    # и сохраним всё эту прелесть в новое изображение и аннотацию.
    for i, roi in enumerate(rois):
        new_boxes = get_roi_anno(roi_box=roi,
                                 obj_boxes=boxes,
                                 target_classes=classes,
                                 ignored_classes=ignor,
                                 rename_as=default_name,
                                 min_overlap=0.6)
        if len(new_boxes)==0:
            print('TRAIN\tВ этой roi {}/{} нет нужных нам объектов'.format(i+1, len(rois)))
            # Значит, в этой roi нет нужных нам объектов
            continue
        # Если мы здесь, значит у нас есть нужные нам объекты и у нас уже даже есть их новые координаты!
        print('TRAIN\tВ этой roi {}/{} есть нужные нам объекты'.format(i + 1, len(rois)))
            # Cначала вырежем и сохраним нашу ОИ
        name = base_name + '_{:02d}_{:02d}'.format(j+1,i+1) # имя без точки и расширения!
        image = cv2.imread(input_images + target)
        if img_d==3:
            image = image[roi[1]:roi[3], roi[0]:roi[2], :]
        if img_d==1:
            image = image[roi[1]:roi[3], roi[0]:roi[2]]
        cv2.imwrite(output_cropped_train + name+'.'+format, image)
            # Теперь сохраним аннотацию
        save_anno_xml(dir=output_annos_train,
                      img_name=name,
                      img_format=format,
                      img_w=img_w,
                      img_h=img_h,
                      img_d=img_d,
                      boxes=new_boxes,
                      quiet=True)

    print('TRAIN\tСохранили новые изображения и аннотации {}/{}'.format(j+1, n))



print('Done!')
