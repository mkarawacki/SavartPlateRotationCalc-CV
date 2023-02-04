# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 19:07:26 2018

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
from scipy import ndimage

MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15
 
 
def alignImages(im1, im2):
 
  # Convert images to grayscale
  #im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  #im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  im1Gray=cv2.cvtColor(im1,cv2.COLOR_GRAY2RGB)
  im2Gray=cv2.cvtColor(im2,cv2.COLOR_GRAY2RGB) 
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  #print(len(matches))
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h
 
 
if __name__ == '__main__':
  
  for i in range(1,7):
    files = [f for f in os.listdir('E:\\DyskGoogle\\TSC90_POL\\20181018\\BD+64106_src') if re.match(r'.*-00'+str(i)+'-p.r.fit', f)]
    print(files)  
    
    
  # Read reference image
  refFilename = "BD+64106-006-p1r.fit"
  print("Reading reference image : ", refFilename)
  imReference = fits.open(refFilename)[0].data#.imread(refFilename, cv2.IMREAD_COLOR)
  '''
  plt.figure()
  plt.imshow(imReference,cmap=plt.cm.gray_r, norm=LogNorm())
  plt.title(refFilename)
  plt.colorbar()
  plt.xlabel('pixel')
  plt.ylabel('pixel')
  '''
  # Read image to be aligned
  imFilename = "BD+64106-006-p3r.fit"
  print("Reading image to align : ", imFilename);  
  im = fits.open(imFilename)[0].data#cv2.imread(imFilename, cv2.IMREAD_COLOR)
  '''
  plt.figure()
  plt.imshow(im,cmap=plt.cm.gray_r, norm=LogNorm())
  plt.title(refFilename)
  plt.colorbar()
  plt.xlabel('pixel')
  plt.ylabel('pixel')
  '''
  hdu=imReference
  ma=hdu.max()
  mi=hdu.min()
  image = np.array(hdu, copy=True)
  '''image.clip(mi,ma, out=image)
  image -=mi
  image //= (ma - mi + 1) / 255.
  '''
  im=image.astype(np.uint8)
  im=cv2.medianBlur(im,5)
  th = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
  cv2.bitwise_not(th,th)
 # cv2.adaptiveThreshold()
  fig=plt.figure()
  plt.imshow(th,cmap=plt.cm.gray_r,norm=LogNorm())
  #img_edges = cv2.Canny(th,127,1000)
  #plt.imshow(img_edges,cmap=plt.cm.gray_r,norm=LogNorm())
  '''
  lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 30)
  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
    '''
  #cv2.imshow("Edges", img_edges)
  #cv2.imshow("Image", im)
  
  im2=np.array(fits.open(imFilename)[0].data,copy=True)
  #im2min=fits.open(imFilename)[0].data.min()
  #im2max=fits.open(imFilename)[0].data.max()
  #im2.clip(min=im2min,max=im2max,out=im2)
  im2=im2.astype(np.uint8)
  
  im2=cv2.medianBlur(im2,5)
  th = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 0)
  cv2.bitwise_not(th,th)
  
  fig=plt.figure()
  plt.imshow(th,cmap=plt.cm.gray_r)
  im2_edges = cv2.Canny(th,127,1000)
  #fig=plt.figure()
  #plt.imshow(im2_edges,cmap=plt.cm.gray_r,norm=LogNorm())
  '''
  lines2 = cv2.HoughLinesP(im2_edges, 1, np.pi / 180.0, 30)
  for line in lines2:
    x1, y1, x2, y2 = line[0]
    cv2.line(im2, (x1, y1), (x2, y2), (0, 255, 0), 3)
  for line in lines:
    x1, y1, x2, y2 = line[0]
    
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
  '''
  #cv2.imshow("Edges", im2_edges)
  #cv2.imshow("Image", im2)  
  '''
  fig, ax = plt.subplots(2, 1, sharex='col', sharey='row')
  ax[0].imshow(im,cmap=plt.cm.gray_r, norm=LogNorm())
  ax[1].imshow(im2,cmap=plt.cm.gray_r, norm=LogNorm())
  plt.title(refFilename)
  #fig.colorbar()
  plt.xlabel('pixel')
  plt.ylabel('pixel')
  
  
  plt.figure()
  plt.imshow(img_edges,cmap=plt.cm.gray_r, norm=LogNorm())
  plt.title(refFilename)
  plt.colorbar()
  plt.xlabel('pixel')
  plt.ylabel('pixel')
  '''
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im2,im)
   
  # Write aligned image to disk. 
  #outFilename = "aligned.jpg"
  #print("Saving aligned image : ", outFilename); 
  #cv2.imwrite(outFilename, imReg)
 
  # Print estimated homography
  #print("Estimated homography : \n",  h)
  theta=-math.atan2(float(h[0,1]),float(h[0,0]))
  print(theta*180.0/math.pi)
  
 # cv2.waitKey(0)
  #cv2.destroyAllWindows()