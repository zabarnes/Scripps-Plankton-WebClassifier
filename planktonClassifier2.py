# -*- coding: utf-8 -*-
"""
Plankton Classifier - updated June 13th, 2014
"""

import numpy as np
import time
from skimage import data, io, filter, morphology, measure, exposure, util
from scipy import fftpack
import json
import os
import sys
import glob
import cv2
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import RFE
from skimage.filter import threshold_otsu
from skimage.feature import greycomatrix, greycoprops
import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
from scipy import interpolate, ndimage
from sklearn import svm
from sklearn import tree
import random, string
from math import pi
sys.path.append('/Users/Orenstein/Documents/python')
from AnnulusProcessing import AnnulusProcessing


class PlanktonImageClassifier:
	
	def __init__(self,name,trainFraction = 0.75, nSelFeatures = 24, sidelobeMin = 0.8):
		self.name = name
		self.X_ = np.array([],ndmin=2)
		self.y_ = np.array([])
		self.clf = svm.SVC(kernel='linear',probability=True)
		#self.clf = tree.DecisionTreeClassifier()
		self.cm = np.array([])
		self.testProbs = np.array([])
		self.sel = np.array([])
		self.trainSel = np.array([])
		self.testSel = np.array([])
		self.trainFraction = trainFraction
		self.trained = False
		self.testPred = np.array([])
		self.dataIsScaled = False
		self.scaler = StandardScaler()
		self.labelId = {}
		self.nFeatures = 50
		self.nSelFeatures = nSelFeatures
		self.sidelobeMin = sidelobeMin
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
		
	def computeSidelobeRatio(self,probs):
		S = np.sort(probs)
		S = S[:,::-1]
		return (S[:,0]-S[:,1])/S[:,0]
		
	def trainAndTest(self):
		# for now always scale
		#if (self.dataIsScaled == False):
		#	self.scaleData()
		
		# select random chunk of data for training. Use the rest to test
		N = len(self.X_)
		TF = int(N*self.trainFraction)
		self.sel = np.random.permutation(N)
		self.trainSel = self.sel[0:TF]
		self.testSel = self.sel[(TF+1):N]
		
		Xtrain = self.X_[self.trainSel,:]
		y_train = self.y_[self.trainSel]
		Xtest = self.X_[self.testSel,:]
		y_test = self.y_[self.testSel]
		
		# Select Features and train
		selector = RFE(svm.SVC(kernel='linear'), step=1, n_features_to_select=self.nSelFeatures)
		selector = selector.fit(Xtrain, y_train)
		self.selFeatures = selector.ranking_==1
		

		self.clf.fit(Xtrain[:,self.selFeatures],y_train)
		self.testPred = self.clf.predict(Xtest[:,self.selFeatures])
		self.testProbs = self.clf.predict_proba(Xtest[:,self.selFeatures])
		
		# Use sidelobe ratio to omit trouble cases
		sl = self.computeSidelobeRatio(self.testProbs)
		inds = sl >= self.sidelobeMin
		
		Xt = Xtest[:,self.selFeatures]
		Xt = Xt[inds,:]
		
		self.testPred = self.clf.predict(Xt)
		self.testProbs = self.clf.predict_proba(Xt)	
		
		self.cm = confusion_matrix(y_test[inds],self.testPred)
		self.score = self.clf.score(Xt,y_test[inds])
		
		self.testIndsAboveSidelobe = inds

		
	def scaleData(self):
		self.scaler.fit(self.X_)
		self.X_ = self.scaler.transform(self.X_)
		self.dataIsScaled = True
	
	# Get a random Id String
	def randomId(self,length):
		return ''.join(random.choice(string.lowercase) for i in range(length))
	
	def classifyImage(self,img):
		
		X = self.extractFeatures(img)
		X = self.scaler.transform(X)
		probs = self.clf.predict_proba(X[self.selFeatures])
		sl = self.computeSidelobeRatio(probs)
		if (sl >= self.sidelobeMin):
			label = self.clf.predict(X[self.selFeatures])
			return label, self.labelId[int(label)]
		else:
			return -1, "-1"
			

	# Extract features from a single image
	def extractFeatures(self,img):
		
		# threshold
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		thresh = threshold_otsu(gray)
		binary = gray >= thresh
		bw_img1 = morphology.closing(binary,morphology.square(3))
		# pad to ensure contour is continuous
		bw_img = np.pad(bw_img1,1, 'constant')

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
		FdSave = np.absolute(FdAll[2:18])/np.absolute(FdAll[1])
		
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
		
		# Get the appropriate FFT padded out to 300 x 300
		freq_simp = fftpack.fftshift(fftpack.fft2(simp, [300, 300]))
		
		# 48 rings, 50 wedges selected from 0 to pi
		blah = AnnulusProcessing(freq_simp, 48, 50) # add number of wedges, etc to init
		rings = blah.make_annular_mean()
		wedges = blah.make_wedge()
		
		# Fill the simplified boundary
		fill = ndimage.binary_fill_holes(simp).astype(int)
		masked = fill * np.pad(gray, 1, 'constant')
		
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
		label_img = morphology.label(bw_img,neighbors=8,background=0)
		features = measure.regionprops(label_img+1)
		
		maxArea = 0
		maxAreaInd = 0
		for f in range(0,len(features)):
		    if features[f].area > maxArea:
		        maxArea = features[f].area
		        maxAreaInd = f
		
		
		# Compute translation, scale, and rotation invariant features
		ii = maxAreaInd
		aspect = features[ii].minor_axis_length/features[ii].major_axis_length
		area1 = features[ii].area/features[ii].convex_area
		area2 = features[ii].area/(features[ii].bbox[3]*features[ii].bbox[2])
		area3 = features[ii].area/(features[ii].perimeter*features[ii].perimeter)
		area4 = area2/area1
		area5 = area3/area1
		fillArea = features[ii].filled_area
		ecc = features[ii].eccentricity
		esd = features[ii].equivalent_diameter
		en = features[ii].euler_number
		sol = features[ii].solidity
		
		# X = np.zeros((self.nFeatures,))
		X = np.zeros(202)
		
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
		X[28:35] = features[ii].moments_hu
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
		
		return X
		
	# Creates a new label by loading images from a directory and
    # processing them. Appends these images to total data matrix
	def addClassDataFromPath(self,imgPath,labelId=-1):
		
		# If no label name is specified define one randomly
		if (labelId == -1):
			labelId = self.randomId(24)
		
		# list images
		imgList = glob.glob(os.path.join(imgPath,'*.png'))
		
		print "computing image descriptors"
		# loop over the images and extract the features for clustering
		#X = np.zeros((len(imgList),self.nFeatures))
		
		X = np.zeros((len(imgList),202))
		ids = []
		rowMap = []
		rowIndex = 0
		count = 0
		for img in imgList:
			bits = img.split('.')
			if (bits[1] != 'png'):
				continue
			# load the color image and convert to gray
			colorImg = cv2.imread(os.path.join(imgPath,img))
			
			# Threshold image and compute descriptors
			X[count,:] = self.extractFeatures(colorImg)
			count = count + 1
			print "image " + str(count) + " of " + str(len(imgList))
		
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
			
		print str(self.X_.shape)
		print str(self.y_.shape)
		
	def displayData(self):
		plt.figure()
		plt.matshow(self.X_)
		plt.title('All Feature Data')
		plt.clim(vmin=-1,vmax=1)
		plt.colorbar()
		plt.ylabel('Index')
		plt.xlabel('Feature Number')

