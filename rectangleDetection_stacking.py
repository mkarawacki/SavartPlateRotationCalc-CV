# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:43:02 2018

@author: Mikołaj
"""

from __future__ import print_function
import cv2
import numpy as np
from astropy.io import fits
import imutils


def PreProcessFITS(file):
  hdu=file
  image = np.array(hdu, copy=True)  
  im=image.astype(np.uint8)
  #im=cv2.GaussianBlur(im,(11,11),0)
  return im 
file_p1='MED_p1.fits'
file_p3='MED_p3.fits'
medf1=fits.open(file_p1)[0].data
medf3=fits.open(file_p3)[0].data

img_orig=medf1
imgrot_orig=medf3

img=PreProcessFITS(img_orig)
imgrot=PreProcessFITS(imgrot_orig)
br=101
sigma=5
blur=cv2.GaussianBlur(img,(br,br),sigma)
blur_rot=cv2.GaussianBlur(imgrot,(br,br),sigma)

ret,thres=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,thres2=cv2.threshold(blur_rot,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

thres_neg=cv2.bitwise_not(thres)
rot_thres_neg=cv2.bitwise_not(thres2)

cnts = cv2.findContours(thres_neg, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cnts, -1, (0,0,255), 3)
dims=[]
centers=[]
angles=[]
rects=[]
boxes=[]
if len(cnts) >0:
    for i in range(0,len(cnts)):
        rect = cv2.minAreaRect(cnts[i])
        c,dim,a=cv2.minAreaRect(cnts[i])
        if dim[0]>1000 and dim[0]<1990 and dim[1]>1000 and dim[1]<1990:
            centers.append(c)
            dims.append(dim)
            angles.append(a)
            rects.append(rect)
            print("Kąt rect: ",a)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)
        else:
            continue
        #prostokat=np.array(box)

        
    cv2.namedWindow('Kontury',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Kontury', 600,600)
    cv2.imshow('Kontury',img)

cnts = cv2.findContours(rot_thres_neg, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cv2.drawContours(cv2.cvtColor(imgrot, cv2.COLOR_GRAY2BGR), cnts, -1, (0,0,255), 3)
dims_rot=[]
centers_rot=[]
angles_rot=[]
rects_rot=[]
boxes_rot=[]
if len(cnts)>0:
    for i in range(0,len(cnts)):
        
        rect_rot = cv2.minAreaRect(cnts[i])
        c_rot,dim_rot,a_rot=cv2.minAreaRect(cnts[i])
        #print("Dim_rot [",i, "] = ",dim_rot)
        #print("a_rot [",i, "] = ",a_rot)
        if dim_rot[0]>1000 and dim_rot[0]<1990 and dim_rot[1]>1000 and dim_rot[1]<1990:
            dims_rot.append(dim_rot)
            centers_rot.append(c_rot)
            angles_rot.append(a_rot)
            
            print("Kąt rect_rot: ", a_rot)
            box_rot = cv2.boxPoints(rect_rot)
            box_rot = np.int0(box_rot)
            cv2.drawContours(imgrot,[box_rot],0,(0,0,255),2)
        else: 
            continue
#pr_rot=np.array(box_rot)

        
print("Względny obrót płytek [*]:",angles_rot[0]-angles[0])        
cv2.namedWindow('Kontury - rot',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Kontury - rot', 600,600)
cv2.imshow('Kontury - rot',imgrot)

cv2.imwrite('Stack_p1.jpg',img)
cv2.imwrite('Stack_p3.jpg',imgrot)

#print(a_rot-a)
#print ("Pole 1= ",dim[0]*dim[1]); print ("Pole 2 = ",dim_rot[0]*dim_rot[1])
#print ("AspectRatio 1=",dim[0]/dim[1]);print ("AspectRatio 2=",dim_rot[0]/dim_rot[1])