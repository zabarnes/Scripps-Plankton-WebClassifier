# -*- coding: utf-8 -*-
"""
Image Processor - updated June 18th, 2014
"""

import numpy as np
import time
from skimage import data, io, filter, morphology, measure, exposure, util, transform, feature
from skimage.filter import threshold_otsu
from skimage.feature import greycomatrix, greycoprops
from scipy import fftpack
import json
import os
import sys
import glob
import cv2
import datetime
import sklearn
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
from scipy import interpolate, ndimage
import random, string
from math import pi, log
from itertools import repeat
from LookUp import look
from AnnulusProcessing import AnnulusProcessing 

class PImageProcessor:
	
	def __init__(self): #name #trainFraction = 0.75, nSelFeatures = 24, sidelobeMin = 0.8):
		#self.name = name
		self.X_ = np.array([],ndmin=2)
		self.y_ = np.array([])
		self.imageData=[]
		self.className = np.array([])
		self.classNum = 0 
		#self.clf = svm.SVC(kernel='linear',probability=True)
		#self.clf = tree.DecisionTreeClassifier()
		self.cm = np.array([])
		self.testProbs = np.array([])
		self.sel = np.array([])
		self.trainSel = np.array([])
		self.testSel = np.array([])
		#self.trainFraction = trainFraction
		self.trained = False
		self.testPred = np.array([])
		self.dataIsScaled = False
		self.scaler = StandardScaler()
		self.labelId = {}
		self.nFeatures = 50
		#self.nSelFeatures = nSelFeatures
		#self.sidelobeMin = sidelobeMin
		self.testIndsAboveSidelobe = np.array([])
		self.selFeatures = []
		self.json = {}
		# for grey level coocurrence matrix
		self.dist = [1,2,4,16,32,64]
		self.ang = [0, pi/4, pi/2, 3*pi / 4]
		self.grey_props = ['contrast', 'homogeneity', 'energy', 'correlation']

		
	def dumpToDisk(self,filepath=''):
		
		if (filepath == ''):
			filepath = './'+"planktonClf_"+str(int(time.time()))+".p"
		pickle.dump( self, open(filepath, "wb" ) )
		
	
	def scaleData(self):
		self.scaler.fit(self.X_)
		self.X_ = self.scaler.transform(self.X_)
		self.dataIsScaled = True
	
	# Get a random Id String
	def randomId(self,length):
		return ''.join(random.choice(string.lowercase) for i in range(length))
			
	# Extract features from a single image
	def extractFeatures(self,img):
		
		# threshold
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		thresh = threshold_otsu(gray)
		binary = gray >= thresh
		bw_img1 = morphology.closing(binary,morphology.square(3))
		# pad to ensure contour is continuous
		bw_img = np.pad(bw_img1, 1, 'constant')

		# plt.imshow(bw_img1)
		# plt.title('Black and White')
		# plt.savefig(self.fnm+"_BW.png",bbox_inches='tight')
		# plt.close()

		# compute intensity histogram features
		gray2 = np.pad(gray,1,'constant')
		pixVals = gray2[bw_img > 0]
		maxPixel = np.max(pixVals)
		minPixel = np.min(pixVals)
		if (maxPixel == 0):
			maxPixel = 1
		
		# normalize histogram
		pixVals = (np.float32(pixVals) - minPixel)/np.max(pixVals)
		histVals = exposure.histogram(pixVals,nbins=64)

		# discrete cosine transform of normalized histogram of pixel values
		allHistFeatures = fftpack.dct(np.float32(histVals[0])) 
		histFeatures = allHistFeatures[1:15]
		
		# Find contours
		contours = measure.find_contours(bw_img,0.5)
		
		# Select largest contour
		maxLength = -1
		maxContour = []
		for cc in contours:
			if (len(cc) > maxLength):
				maxLength = len(cc)
				maxContour = cc

		# fig, ax = plt.subplots()
		# #ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)
		# for n, contour in enumerate(contours):
		# 	ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
		# ax.axis('image')
		# plt.title("Contours")
		# plt.savefig(self.fnm+"_C.png",bbox_inches='tight')
		# plt.close()
		
		# Represent contour in fourier space. Make scale invarient by 
		# dividing by DC term. Can make rotation invariant by subtracting 
		# phase of first term		

		# Interpolate to 4096 point contour
		interpX = interpolate.interp1d(range(0,maxLength),maxContour[:,0])
		interpY = interpolate.interp1d(range(0,maxLength),maxContour[:,1])
		newS = np.linspace(0,maxLength-1,4096)
		cX = interpX(newS)
		cY = interpY(newS)
		cPath = cX +1j*cY
		FdAll = np.fft.fft(cPath)
		FdSave = np.log(np.absolute(FdAll[2:18])/np.absolute(FdAll[1]))
		
		# Simplify the boundary
		cen = np.fft.fftshift(FdAll)
		
		# take first 10% of fourier coefficents
		cen2 = np.hstack([np.zeros(1843), cen[1843:2253], np.zeros(1843)])
		# Back project to simplified boundary
		back = np.fft.ifft(np.fft.ifftshift(cen2))
		
		xx = np.round(back.real)
		yy = np.round(back.imag)
		
		m = bw_img.shape[0]
		n = bw_img.shape[1]
		
		xx = xx.astype(np.int)
		yy = yy.astype(np.int)
		
		simp = np.zeros([m,n])
		simp[xx,yy] = 1

		# fig, ax = plt.subplots()
		# ax.imshow(img)

		# ax.plot(maxContour[:, 1], maxContour[:, 0], linewidth=2)
		# ax.axis('image')
		# plt.title("Max Contour")
		# plt.savefig(self.fnm+"_MC.png",bbox_inches='tight')
		# plt.close()
		
		# Get the appropriate FFT padded out to 300 x 300
		freq_simp = fftpack.fftshift(fftpack.fft2(simp, [300, 300]))

		# 48 rings, 50 wedges selected from 0 to pi
		ann = AnnulusProcessing(freq_simp, 48, 50) # add number of wedges, etc to init
		rings = ann.make_annular_mean()
		wedges = ann.make_wedge()

		
		# Fill the simplified boundary
		fill = ndimage.binary_fill_holes(simp).astype(int)
		masked = fill * np.pad(gray, 1, 'constant')

		# plt.imshow(masked)
		# plt.title("Masked")
		# plt.savefig(self.fnm+"_M.png",bbox_inches='tight')
		# plt.close()


		# Gray level coocurrence matrix 
		P = greycomatrix(masked, distances = self.dist, angles = self.ang, normed = True)
		grey_mat = np.zeros([24,2]) 
		flag = 0
		for name in self.grey_props:
		    stat = greycoprops(P, name)
		    grey_mat[flag:flag+6,0] = np.mean(stat,1)
		    grey_mat[flag:flag+6,1] = np.std(stat,1)
		    flag += 6
		
		# Texture descripters
		prob = np.histogram(masked, 256) # assume gray scale with 256 levels
		prob = np.asarray(prob[0])
		prob[0] = 0 # don't count black pixels
		prob = prob / prob.sum()
		vec = np.arange(0, len(prob)) / (len(prob) - 1)
		ind = np.nonzero(prob)[0]
		
		# mean grey value
		mu = np.sum(vec[ind] * prob[ind])
		
		# variance 
		var = np.sum((((vec[ind] - mu)**2) * prob[ind]))
		
		# standard deviation
		std =  np.sqrt(var)
		
		# contrast
		cont = 1 - 1/(1 + var) 	
		
		# 3rd moment
		thir = np.sum(((vec[ind] - mu)**3)*prob[ind])	
		
		# Uniformity
		uni = np.sum(prob[0]**2)
		
		# Entropy
		ent = - np.sum(prob[ind] * np.log2(prob[ind]))
		
		# Compute morphological descriptors
		label_img = measure.label(bw_img,neighbors=8,background=0)
		features = measure.regionprops(label_img+1)
		
		maxArea = 0
		maxAreaInd = 0
		for f in range(0,len(features)):
		    if features[f].area > maxArea:
		        maxArea = features[f].area
		        maxAreaInd = f
		
		
		# Compute translation, scal and rotation invariant features
		ii = maxAreaInd
		aspect = features[ii].minor_axis_length/features[ii].major_axis_length
		area1 = features[ii].area/features[ii].convex_area
		area2 = features[ii].area/(features[ii].bbox[3]*features[ii].bbox[2])
		area3 = features[ii].area/(features[ii].perimeter*features[ii].perimeter)
		area4 = area2/area1
		area5 = area3/area1
		per = features[ii].perimeter
		simp_area = features[ii].area 
		pa = per/simp_area
		fillArea = features[ii].filled_area
		ecc = features[ii].eccentricity
		esd = features[ii].equivalent_diameter
		en = features[ii].euler_number
		sol = features[ii].solidity
		momC= features[ii].moments_central
		ext = features[ii].extent

		# copeTestImg = cv2.imread("copeTest.png")
		# copegray = cv2.cvtColor(copeTestImg,cv2.COLOR_BGR2GRAY)
		# copethresh = threshold_otsu(copegray)
		# copebinary = copegray >= copethresh
		# cope_img1 = morphology.closing(copebinary,morphology.square(3))		# pad to ensure contour is continuous
		# cope_img = np.pad(cope_img1, 1, 'constant')
		# size=bw_img.shape

		# copeTestImgRes = transform.resize(cope_img,size, mode='nearest')
		# intarray=np.around(copeTestImgRes)
		# intarray = intarray.astype(dtype="uint8")
		# copeTestImgRot = transform.rotate(cope_img,features[ii].orientation).astype(dtype="uint8")
		# copeTestImgBoth = transform.rotate(copeTestImgRes, features[ii].orientation).astype(dtype="uint8")


		# SS_res = measure.structural_similarity(bw_img,intarray)
		# SS_both = measure.structural_similarity(bw_img,copeTestImgBoth)
		

		# MT = feature.match_template(bw_img,copeTestImgBoth)
		# maxMT = np.amax(MT)


		#likelyT=measure.structural_similarity(bw_img,copeTestImg)
		
		X = np.zeros(212)
		
		X[0:16] = FdSave
		X[17] = aspect
		X[18] = area1
		X[19] = area2
		X[20] = area3
		X[21] = area4
		X[22] = area5
		X[23] = fillArea
		X[24] = ecc
		X[25] = esd
		X[26] = en
		X[27] = sol
		X[28:35] = np.log(features[ii].moments_hu)
		X[36:50] = histFeatures
		X[50:98] = rings # only use first 10?
		X[98:148] = wedges # sort these
		X[148:172] = grey_mat[:,0]
		X[172:196] = grey_mat[:,1]
		X[196] = mu
		X[197] = std
		X[198] = cont
		X[199] = thir
		X[200] = uni
		X[201] = ent
		X[202] = per
		X[203] = simp_area
		X[204] = pa
		X[205] = ext
		# X[206] = np.log(features[ii].inertia_tensor_eigvals[0])
		# X[207] = np.log(features[ii].inertia_tensor_eigvals[1])
		# X[208] = features[ii].orientation
		# X[209] = maxMT
		# X[210] = SS_res
		# X[211] = SS_both
		
		return X 

	def imgFromPath(self, imgPath):
		return glob.glob(os.path.join(imgPath,'*.png'))

	def imgReadIn(self, imgPath, img):
		return cv2.imread(os.path.join(imgPath,img))

	def addImagesForProc(self, imgPath):
		imgList = self.imgFromPath(imgPath)
		
		print "Computing features for unclassified images"
		# loop over the images and extract the features for clustering
		
		X = np.zeros((len(imgList),212))
		count = 0
		for img in imgList:
			bits = img.split('.')
			if (bits[1] != 'png'):
				continue
			# load the color image and convert to gray
			colorImg = self.imgReadIn(imgPath,img)

			# Threshold image and compute descriptors
			X[count,:]=self.extractFeatures(colorImg)
			count = count + 1
			print "Image " + str(count) + " of " + str(len(imgList)) +" ready to classify"

		self.imageData=X

	def processImg(self,imgPath):
		#read in image from path and process and return value
		colorImg = cv2.imread(imgPath)
		return self.extractFeatures(colorImg)


	def addClass(self,imgPath,labelId=-1):
		
		# If no label name is specified define one randomly
		if (labelId == -1):
			labelId = self.randomId(24)
		
		# list images
		imgList = self.imgFromPath(imgPath)
		
		print "Computing class parameters for (put class id here)"
		# loop over the images and extract the features for clustering
		
		X = np.zeros((len(imgList),212))
		temp=[]
		fAvg = np.zeros((1,212))
		fVar = np.zeros((1,212))
		ids = []
		rowMap = []
		rowIndex = 0
		count = 0
		for img in imgList:
			bits = img.split('.')
			if (bits[1] != 'png'):
				continue
			# load the color image and convert to gray
			# self.fnm=look.classes(count)
			colorImg = self.imgReadIn(imgPath,img)
			# plt.imshow(colorImg)
			# plt.title("Color Image")
			# plt.savefig(self.fnm+"_CI.png",bbox_inches='tight')
			# plt.close()

			

			# Threshold image and compute descriptors
			temp=self.extractFeatures(colorImg)
			X[count,:]=temp
			# fAvg=np.add(fAvg,temp)
			# fVar=np.add(fVar,np.power(temp,2))
			count = count + 1
			print "Image " + str(count) + " of " + str(len(imgList))

		# fAvg=np.divide(fAvg,len(imgList))
		# fVar=np.subtract(np.divide(fVar,len(imgList)),np.power(fAvg,2)) 
		# fStats=np.concatenate((fAvg,fVar))

		#self.classStats[self.classNum]=fStats
		self.classNum = self.classNum+1

		# append data and create new label
		if (len(self.y_) == 0):
			labelNum = 0
			self.labelId = [labelId]
		else:
			labelNum = np.max(self.y_)+1
			self.labelId = self.labelId + [labelId]
		
		newLabels = labelNum*np.ones((count,))
		
		if (len(self.y_) == 0):
			self.X_ = X
			self.y_ = newLabels
			
		else:
			self.X_ = np.append(self.X_,X,0)
			self.y_ = np.append(self.y_,newLabels,0)



