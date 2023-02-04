<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:26:19 2018

@author: Mikołaj
"""

from __future__ import print_function
import cv2
import numpy as np
from astropy.io import fits
import time
import os,re
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import imutils
from scipy import ndimage


def PreProcessFITS(file):
  hdu=file
  ma=hdu.max()
  mi=hdu.min()
  image = np.array(hdu, copy=True)
  image.clip(mi,ma, out=image)
  image -=mi
  #image //= (ma - mi + 1) / 255.
  
  im=image.astype(np.uint8)
  im=cv2.GaussianBlur(im,(101,101),20)
  #im=cv2.inRange(im,lower,upper)
  #cv2.subtract(image.astype(np.uint8),im)
  #im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
  #im=cv2.threshold(im,127,255,0)
  #cv2.bitwise_not(im,im)
  #cv2.subtract(im,image.astype(np.uint8))
  return im 
lower =0# np.array([0, 0, 0])
upper =30# np.array([15, 15, 15])
linie=[] 
for i in range(1,7):
    files = [f for f in os.listdir('E:\\DyskGoogle\\TSC90_POL\\20181018\\BD+64106_src') if re.match(r'.*-00'+str(i)+'-p.r.fit', f)]
    print(files)

    img = fits.open(files[0])[0].data
    imgrot = fits.open(files[1])[0].data
    
    img_orig=img
    imgrot_orig=imgrot
    #add_orig=cv2.add(img_orig,img)
    img=PreProcessFITS(img)
    imgrot=PreProcessFITS(imgrot)
    
    #add=cv2.addWeighted(img,0.8,img_orig.astype(np.uint8),0.2,0)#cv2.subtract(img,img_orig.astype(np.uint8))
   
   
    shapeMask = cv2.inRange(img, lower, upper)
    shapeMask_rot = cv2.inRange(imgrot, lower, upper)
    
    #add_rot=cv2.addWeighted(shapeMask_rot,0.8,imgrot_orig.astype(np.uint8),0.2,0)
    shapeMask_rot=cv2.bitwise_not(shapeMask_rot)
    shapeMask=cv2.bitwise_not(shapeMask)
    
    
    
    fig,a=plt.subplots(2,3, sharex=True, sharey=True)
    fig.suptitle(files[0][:-8])
    a[0,0].imshow(img_orig,cmap=plt.cm.gray_r, norm=LogNorm())
    a[0,1].imshow(img,cmap=plt.cm.gray, norm=LogNorm())
    a[0,2].imshow(shapeMask,cmap=plt.cm.gray)
    a[1,0].imshow(imgrot_orig,cmap=plt.cm.gray_r, norm=LogNorm())
    a[1,1].imshow(imgrot,cmap=plt.cm.gray, norm=LogNorm())
    a[1,2].imshow(shapeMask_rot,cmap=plt.cm.gray)
    #plt.savefig("Porównanie "+files[0][:-5]+"-"+files[1][:-5]+".png")
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines=cv2.HoughLinesP(img,1,np.pi/180,1000,1750,500)
    #for i in range(0,4):
    #if len(lines)>0:
# =============================================================================
#         for rho,theta in lines[0]:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a*rho
#             y0 = b*rho
#             x1 = int(x0 + 1000*(-b))
#             y1 = int(y0 + 1000*(a))
#             x2 = int(x0 - 1000*(-b))
#             y2 = int(y0 - 1000*(a))
# =============================================================================
    #    kopiamaski=shapeMask.copy()
    #    for x1,y1,x2,y2 in lines[0]:
    #        cv2.line(kopiamaski,(x1,y1),(x2,y2),(0,255,0),2)
            #cv2.line(shapeMask,(x1,y1),(x2,y2),(0,0,255),2)
    #linie.append(((x1,y1),(x2,y2)))
    
    #linie.append(lines)
    #cv2.imwrite('houghlines-'+files[0][:-4]+'.jpg',kopiamaski)    
    
    #print("Ilosc linii: ",len(linie[i-1]))#, "Kształt: ", linie.shape())
    cnts = cv2.findContours(shapeMask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("I found {} black shapes".format(len(cnts)))
    cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cnts, -1, (0,0,255), 3)
    M = cv2.moments(cnts[0])
    #print(M)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print("Cx= ",cx," Cy= ",cy)
    
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    for c in cnts:
        cv2.drawContours(shapeMask,[c],0,(0,0,255),2)
    cv2.namedWindow('Kontury',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Kontury', 600,600)
    cv2.imshow('Kontury',shapeMask)
    #print(box)
    #plt.imshow(add,cmap=plt.cm.gray_r, norm=LogNorm())
    #plt.title("Suma - "+ files[0][:-4])
    '''
    fig2, ax2 = plt.subplots(2, 1, sharex='col', sharey='row')
    plt.title(files[1])
    ax2[0].imshow(img,cmap=plt.cm.gray_r, norm=LogNorm())
    ax2[1].imshow(imgrot,cmap=plt.cm.gray_r, norm=LogNorm())
    plt.savefig("Porównanie"+files[0][:-4]+"-"+files[1][:-4]+".png")
    #ret,thresh = cv2.threshold(img,127,255,0)
    outimg,contours,hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    #print(M)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #print("Cx= ",cx," Cy= ",cy)
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    #print(box)
    ret,thresh = cv2.threshold(imgrot,127,255,0)
    outimg,contours,hierarchy = cv2.findContours(imgrot, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    cntrot = contours[0]
    Mrot = cv2.moments(cntrot)
    #print(Mrot)
    
    #cx = int(Mrot['m10']/Mrot['m00'])
    #cy = int(Mrot['m01']/Mrot['m00'])
    #print("Cx= ",cx," Cy= ",cy)
    rectrot = cv2.minAreaRect(cntrot)
    boxrot = cv2.boxPoints(rectrot)
    boxrot = np.int0(boxrot)
    #cv2.drawContours(imgrot,[box],0,(0,0,255),2)
    #print(boxrot)
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:26:19 2018

@author: Mikołaj
"""

from __future__ import print_function
import cv2
import numpy as np
from astropy.io import fits
import time
import os,re
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import imutils
from scipy import ndimage


def PreProcessFITS(file):
  hdu=file
  ma=hdu.max()
  mi=hdu.min()
  image = np.array(hdu, copy=True)
  image.clip(mi,ma, out=image)
  image -=mi
  #image //= (ma - mi + 1) / 255.
  
  im=image.astype(np.uint8)
  im=cv2.GaussianBlur(im,(101,101),20)
  #im=cv2.inRange(im,lower,upper)
  #cv2.subtract(image.astype(np.uint8),im)
  #im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
  #im=cv2.threshold(im,127,255,0)
  #cv2.bitwise_not(im,im)
  #cv2.subtract(im,image.astype(np.uint8))
  return im 
lower =0# np.array([0, 0, 0])
upper =30# np.array([15, 15, 15])
linie=[] 
for i in range(1,7):
    files = [f for f in os.listdir('E:\\DyskGoogle\\TSC90_POL\\20181018\\BD+64106_src') if re.match(r'.*-00'+str(i)+'-p.r.fit', f)]
    print(files)

    img = fits.open(files[0])[0].data
    imgrot = fits.open(files[1])[0].data
    
    img_orig=img
    imgrot_orig=imgrot
    #add_orig=cv2.add(img_orig,img)
    img=PreProcessFITS(img)
    imgrot=PreProcessFITS(imgrot)
    
    #add=cv2.addWeighted(img,0.8,img_orig.astype(np.uint8),0.2,0)#cv2.subtract(img,img_orig.astype(np.uint8))
   
   
    shapeMask = cv2.inRange(img, lower, upper)
    shapeMask_rot = cv2.inRange(imgrot, lower, upper)
    
    #add_rot=cv2.addWeighted(shapeMask_rot,0.8,imgrot_orig.astype(np.uint8),0.2,0)
    shapeMask_rot=cv2.bitwise_not(shapeMask_rot)
    shapeMask=cv2.bitwise_not(shapeMask)
    
    
    
    fig,a=plt.subplots(2,3, sharex=True, sharey=True)
    fig.suptitle(files[0][:-8])
    a[0,0].imshow(img_orig,cmap=plt.cm.gray_r, norm=LogNorm())
    a[0,1].imshow(img,cmap=plt.cm.gray, norm=LogNorm())
    a[0,2].imshow(shapeMask,cmap=plt.cm.gray)
    a[1,0].imshow(imgrot_orig,cmap=plt.cm.gray_r, norm=LogNorm())
    a[1,1].imshow(imgrot,cmap=plt.cm.gray, norm=LogNorm())
    a[1,2].imshow(shapeMask_rot,cmap=plt.cm.gray)
    #plt.savefig("Porównanie "+files[0][:-5]+"-"+files[1][:-5]+".png")
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    lines=cv2.HoughLinesP(img,1,np.pi/180,1000,1750,500)
    #for i in range(0,4):
    #if len(lines)>0:
# =============================================================================
#         for rho,theta in lines[0]:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a*rho
#             y0 = b*rho
#             x1 = int(x0 + 1000*(-b))
#             y1 = int(y0 + 1000*(a))
#             x2 = int(x0 - 1000*(-b))
#             y2 = int(y0 - 1000*(a))
# =============================================================================
    #    kopiamaski=shapeMask.copy()
    #    for x1,y1,x2,y2 in lines[0]:
    #        cv2.line(kopiamaski,(x1,y1),(x2,y2),(0,255,0),2)
            #cv2.line(shapeMask,(x1,y1),(x2,y2),(0,0,255),2)
    #linie.append(((x1,y1),(x2,y2)))
    
    #linie.append(lines)
    #cv2.imwrite('houghlines-'+files[0][:-4]+'.jpg',kopiamaski)    
    
    #print("Ilosc linii: ",len(linie[i-1]))#, "Kształt: ", linie.shape())
    cnts = cv2.findContours(shapeMask, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("I found {} black shapes".format(len(cnts)))
    cv2.drawContours(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cnts, -1, (0,0,255), 3)
    M = cv2.moments(cnts[0])
    #print(M)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print("Cx= ",cx," Cy= ",cy)
    
    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    for c in cnts:
        cv2.drawContours(shapeMask,[c],0,(0,0,255),2)
    cv2.namedWindow('Kontury',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Kontury', 600,600)
    cv2.imshow('Kontury',shapeMask)
    #print(box)
    #plt.imshow(add,cmap=plt.cm.gray_r, norm=LogNorm())
    #plt.title("Suma - "+ files[0][:-4])
    '''
    fig2, ax2 = plt.subplots(2, 1, sharex='col', sharey='row')
    plt.title(files[1])
    ax2[0].imshow(img,cmap=plt.cm.gray_r, norm=LogNorm())
    ax2[1].imshow(imgrot,cmap=plt.cm.gray_r, norm=LogNorm())
    plt.savefig("Porównanie"+files[0][:-4]+"-"+files[1][:-4]+".png")
    #ret,thresh = cv2.threshold(img,127,255,0)
    outimg,contours,hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    M = cv2.moments(cnt)
    #print(M)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #print("Cx= ",cx," Cy= ",cy)
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #cv2.drawContours(img,[box],0,(0,0,255),2)
    #print(box)
    ret,thresh = cv2.threshold(imgrot,127,255,0)
    outimg,contours,hierarchy = cv2.findContours(imgrot, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    cntrot = contours[0]
    Mrot = cv2.moments(cntrot)
    #print(Mrot)
    
    #cx = int(Mrot['m10']/Mrot['m00'])
    #cy = int(Mrot['m01']/Mrot['m00'])
    #print("Cx= ",cx," Cy= ",cy)
    rectrot = cv2.minAreaRect(cntrot)
    boxrot = cv2.boxPoints(rectrot)
    boxrot = np.int0(boxrot)
    #cv2.drawContours(imgrot,[box],0,(0,0,255),2)
    #print(boxrot)
>>>>>>> ce87373 (Initial commit)
    '''