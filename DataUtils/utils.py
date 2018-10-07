#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import shutil
import random
from xml.etree.ElementTree import Element, ElementTree, SubElement, parse

def getFormat(name):
    # Берет на вход имя,
    # возвращает подстроку-формат и позицию с конца (отрицательное число)
    # БЕЗ ТОЧКИ
    k = name.rfind('.')
    name = name[k + 1:]
    k = -(len(name))
    return name, k

def get_roi_image(image_w, image_h, number, target_w=608, target_h=608, horiz_offset=(-100,100)):
    # Функция выбора ОИ на изображении.
    # Принимает на вход размеры изображения, целевые размеры, и некоторые параметры модификации
    # Возвращает флаг правильности исполнения и список рамок с ОИ
    #
    # Имеем опорные значения - точка центра ОИ и её возможные отклонения,
    # и размеры (ширина, высота) ОИ и их возможные отклонения

    # Координаты центра. Координата hc изменяться не будет
    xc = image_w // 2 + image_w % 2
    yc = image_h // 2 + image_h % 2

    if target_w == target_h:
        if image_h < image_w:
            max_h = image_h
            max_w = image_h
        else:
            max_h = image_w
            max_w = image_w
    else:
        print('Пока что можно обрезать только квадратиком, простите!')
        return False, None

    min_h = target_h
    min_w = target_w

    offset_l, offset_r = horiz_offset
    if offset_l == 0 and offset_r == 0:
        print('При таких параметрах сдвига можем сделать только один кадр!')
        number = 1
    if offset_l > 0 or offset_r < 0:
        print('Некорректные значения сдвига!')
        return False, None
    offset_l = xc + int(np.round(float(offset_l) / 100.0 * (image_w - max_w) / 2))
    offset_r = xc + int(np.round(float(offset_r) / 100.0 * (image_w - max_w) / 2))

    # Здесь уже рандомом выбираем конкретные ОИ в количестве number
    delta = (offset_r-offset_l)/number
    roi_cx = [random.randint(offset_l + int(delta*n), offset_l + int(delta*(n+1))) for n in range(number)]
    roi_cy = [yc    for n in range(number)]
        # Это изменится, если реализую zoom
    roi_w  = [max_w for n in range(number)]
    roi_h  = [max_h for n in range(number)]
    if not len(roi_cx)==len(roi_w)==len(roi_h)==len(roi_cy):
        print('Размеры и значения положения созданы для разного числа вырезок!')
        return False, None

    # Составим итоговые координаты ОИ
    boxes = []
    for cx, cy, w, h in zip(roi_cx, roi_cy, roi_w, roi_h):
        x1 = cx-(w//2)
        x2 = cx+(w//2)
        y1 = cy-(h//2)
        y2 = cy+(h//2)
        boxes.append((x1, y1, x2, y2))

    return True, boxes

def intersection(roi, obj):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    obj_x1, obj_y1, obj_x2, obj_y2 = obj
    w = min(roi_x2, obj_x2) - max(roi_x1, obj_x1)
    h = min(roi_y2, obj_y2) - max(roi_y1, obj_y1)
    res = 0.
    if not ((w < 0) or (h < 0)):
        res = w*h / ((obj_x2-obj_x1)*(obj_y2-obj_y1))
    return res

def getObjCoordsInROI(roi_box, obj_box, threshold=1.):
    # Проверяем, входит ли объект в ОИ, и если входит,
    # возвращаем новые координаты объекта. Если не входит,
    # возвращаем None

    # Проверим, сколько процентов объекта остается в ОИ
    part = intersection(roi=roi_box, obj=obj_box)
    if part < threshold:
        return None
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
        obj_x1, obj_y1, obj_x2, obj_y2 = obj_box
        obj_x1 = max(obj_x1, roi_x1) - roi_x1
        obj_y1 = max(obj_y1, roi_y1) - roi_y1
        obj_x2 = min(obj_x2, roi_x2) - roi_x1
        obj_y2 = min(obj_y2, roi_y2) - roi_y1
        return (obj_x1, obj_y1, obj_x2, obj_y2)

def get_roi_anno(roi_box, obj_boxes, target_classes=(), ignored_classes=(), rename_as='', min_overlap=1):
    # Принимаем на вход размеры оригинального изображения,
    # 4 координаты (x1, y1, x2, y2) ОИ,
    # и список с данными объектов tuple([confidence(float),] [objid(str),] classname(str), x1, y1, x2, y2).
    # Возвращаем новый список с данными объектов
    # с координатами уже относительно ОИ
    new_obj_boxes = []
    if len(target_classes)>0 and len(ignored_classes)>0:
        intersect = (set(target_classes) - set(ignored_classes)) + (set(ignored_classes) - set(target_classes))
        assert len(intersect)==0, 'target_classes и ignored_classes не должны пересекаться!'

    for obj_box in obj_boxes:
        # Скопируем бокс, чтобы не менять оригинальный
        new_obj_box = list(obj_box)
        # ФИЛЬТРАЦИЯ
            # если класс объекта в числе игнорируемых, отбросим
        if len(ignored_classes)>0 and     (new_obj_box[-5] in ignored_classes):
            continue
            # если этот класс не в числе игнорируемых или целевых, то переименуем его, если есть имя, иначе отбросим
        if len(target_classes)>0  and not (new_obj_box[-5] in target_classes):
            if rename_as:
                new_obj_box[-5] = rename_as
            else:
                continue
            # если целевые классы не заданы, но есть дефолтное имя, то переименуем и отправим в список объектов
        if len(target_classes)==0 and     rename_as:
            new_obj_box[-5] = rename_as
        # Проверяем, входит ли оъект в ОИ, и если да, какие у него будут относительные координаты
        res = getObjCoordsInROI(roi_box=roi_box, obj_box=new_obj_box[-4:], threshold=min_overlap)
        if res:
            # если входит, добавляем в иоговый список для сохранения в аннотациях
            new_obj_box[-4] = res[0]
            new_obj_box[-3] = res[1]
            new_obj_box[-2] = res[2]
            new_obj_box[-1] = res[3]
            new_obj_boxes.append(new_obj_box)

    return new_obj_boxes

def parse_annotation_xml(anno_path):
    # Вид каждого элемента в списке boxes:
    # tuple([confidence(float),] [objid(str),] classname(str), x1, y1, x2, y2).
    img_w = 0
    img_h = 0
    imgname = ''
    boxes = []

    tree = parse(anno_path)

    for elem in tree.iter():
        if 'filename' in elem.tag:
            imgname = elem.text
        if 'width' in elem.tag:
            img_w = int(elem.text)
        if 'height' in elem.tag:
            img_h = int(elem.text)
        if 'depth' in elem.tag:
            img_d = int(elem.text)
        if 'object' in elem.tag or 'part' in elem.tag:
            box = ['', '', '', 0, 0, 0, 0]

            for attr in list(elem):
                if 'confidence' in attr.tag:
                    box[0] = float(attr.text)
                if 'id' in attr.tag:
                    box[1] = int(round(float(attr.text)))
                if 'name' in attr.tag:
                    box[2] = attr.text

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            box[3] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            box[4] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            box[5] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            box[6] = int(round(float(dim.text)))
            if box[2] and (box[5] > 0) and (box[6] > 0):
                # Если id и confidence так и не было заполнено, то удалим их
                if not box[1]:
                    del box[1]
                if not box[0]:
                    del box[0]
                boxes.append(box)

    return imgname, img_w, img_h, img_d, boxes

def save_anno_xml(dir,
              img_name,
              img_format,
              img_w,
              img_h,
              img_d,
              boxes,
              quiet=False,
              minConf=-1.):

    '''
    boxes:      tuple([confidence(float),] [objid(str),] classname(str), x1, y1, x2, y2)
    minConf:    минимально допустимое значение уверенности, которое должно быть у бокса.
                если = -1., значит, у боксов вообще нет его.
    '''
    # Сохраним новую аннотацию
    annotation = Element('annotation')
    filename = SubElement(annotation, "filename")
    filename.text = img_name + '.' + img_format
    size = SubElement(annotation, "size")
    width = SubElement(size, "width")
    height = SubElement(size, "height")
    depth = SubElement(size, "depth")
    width.text = str(img_w)
    height.text = str(img_h)
    depth.text = str(img_d)
    if len(boxes) > 0:
        for box in boxes:
            conf  = -1.
            objid = -1
            if minConf >= 0:
                if len(box) == 6:
                    conf, classname, x1, y1, x2, y2 = box
                elif len(box) == 7:
                    conf, objid, classname, x1, y1, x2, y2 = box
                else:
                    print('Параметр boxes передан в неправильном формате!')
                    return
                # Фильтр по уверенности
                if conf < minConf:
                    continue

            else:
                if len(box) == 5:
                    classname, x1, y1, x2, y2 = box
                elif len(box) == 6:
                    objid, classname, x1, y1, x2, y2 = box
                else:
                    print('Параметр boxes передан в неправильном формате!')
                    return

            object = SubElement(annotation, "object")
            name = SubElement(object, "name")
            name.text = classname
            if int(objid) >= 0:
                id = SubElement(object, "id")
                id.text = str(objid)
            if minConf >= 0.:
                confidence = SubElement(object, "confidence")
                confidence.text = str(conf)
            bndbox = SubElement(object, "bndbox")
            xmin = SubElement(bndbox, "xmin")
            ymin = SubElement(bndbox, "ymin")
            xmax = SubElement(bndbox, "xmax")
            ymax = SubElement(bndbox, "ymax")
            xmin.text = str(x1)
            ymin.text = str(y1)
            xmax.text = str(x2)
            ymax.text = str(y2)
    et = ElementTree(annotation)
    et.write(dir + img_name + '.xml')
    if not quiet:
        print("File {} has been saved".format(img_name))

def aug_resize(img, boxes, number, h_scale=(0.8, 0.95), w_scale=(0.8, 0.95)):
    # Меняем пропорции изображения и данные для аннотации
    print('')

