#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import shutil
from xml.etree.ElementTree import Element, ElementTree, SubElement
import csv

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

def imgdata_to_xml(data, output_path, img_format, imgw = 1280, imgh = 720, newname = ''):
    # Здесь принимаем список аннотаций, где в каждом элементе списка
    # (filename, bboxes), bboxes - список, в каждом элементе которого
    # (x1, y1, x2, y2)
    #

    if not img_format:
        print('Нужно задать формат изображений!')
        return

    imgname, bboxes = data
    #
    imgname = newname if newname else imgname[:-len(img_format)-1]
    imgname = imgname + '.' + img_format
    #
    annotation = Element('annotation')
    filename = SubElement(annotation, "filename")
    filename.text = imgname
    size = SubElement(annotation, "size")
    width  = SubElement(size, "width")
    height = SubElement(size, "height")
    depth  = SubElement(size, "depth")
    width.text  = str(imgw)
    height.text = str(imgh)
    depth.text  = '3'
    if len(bboxes) > 0:
        for box in bboxes:
            if len(box) == 5:
                x1, y1, x2, y2, classname = box
            if len(box) == 6:
                x1, y1, x2, y2, classname = box
            object = SubElement(annotation, "object")
            name = SubElement(object, "name")
            name.text = classname
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
    newname = newname + '.xml' if newname else imgname[:-4] + '.xml'
    et.write(output_path + newname)
    print("File {} has been saved".format(newname))



