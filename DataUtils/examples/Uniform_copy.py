#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import random
from utils import getFormat


fromdir = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/4-SuperSign/val/'
todir   = '/media/data/ObjectDetectionExperiments/Datasets/5_RTSD/CustomSets/4-SuperSign/test/'
#names   = '/media/data/ObjectDetectionExperiments/Datasets/4_LISA/2_extended_set/annotations/'
N = 2000

names = os.listdir(fromdir+'images/')
n = len(names)
if n < N:
    N = n

names = random.sample(names, N)
format, negposition = getFormat(names[0])
namepairs = [(name, name[:negposition] + 'xml') for name in names]

for i,namepair in enumerate(namepairs):
    shutil.copy(fromdir+'images/'+namepair[0], todir+'images/'+namepair[0])
    shutil.copy(fromdir + 'annotations/' + namepair[1], todir + 'annotations/' + namepair[1])
    print('Скопировали изображение-аннотацию {}/{}'.format(i+1, N))
