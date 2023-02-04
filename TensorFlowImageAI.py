<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:46:10 2018

@author: Mikołaj
"""
from __future__ import print_function
from imageai.Detection import ObjectDetection

import cv2
import numpy as np
from astropy.io import fits
import time
import os,re
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from scipy import ndimage
execution_path = os.getcwd()
files=['BD+64106-001-p1r.fit', 'BD+64106-001-p3r.fit']
img = fits.open(files[0])[0].data
imgrot = fits.open(files[1])[0].data

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=img, output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 21:46:10 2018

@author: Mikołaj
"""
from __future__ import print_function
from imageai.Detection import ObjectDetection

import cv2
import numpy as np
from astropy.io import fits
import time
import os,re
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
from scipy import ndimage
execution_path = os.getcwd()
files=['BD+64106-001-p1r.fit', 'BD+64106-001-p3r.fit']
img = fits.open(files[0])[0].data
imgrot = fits.open(files[1])[0].data

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=img, output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
>>>>>>> ce87373 (Initial commit)
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )