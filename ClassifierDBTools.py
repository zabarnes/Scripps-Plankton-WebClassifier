import sys
import numpy as np
import json
import os
import pymongo
from pymongo import MongoClient
import cPickle as pickle
import datetime
import MachineLearnTools
from bson.objectid import ObjectId
from MachineLearnTools import NaiveBayes
from ImProc import PImageProcessor as ImProc

class TrainPhase:

	def __init__(self, ip='localhost', port='27017', debug=False):
		#classifier variables
		self.classCollectionName = 'classifiers'
		self.imgCollectionName = 'planktonDB'
		self.taxaCollectionName = 'taxaTags'

		#port and ip
		self.ip = ip
		self.port = port

		#connect to db
		client = MongoClient(self.ip,int(self.port))
		#self.db = client.planktonDB
		self.db = client.phytoDB

		#connect to each important collection
		self.classCollection = self.db[self.classCollectionName]
		self.imgCollection = self.db[self.imgCollectionName]
		self.taxaCollection = self.db[self.taxaCollectionName]

		#initialize other variables
		self.limitToUpdate = 100 #arbitrary assertion
		self.nbSelectedFeatures = [0,1,2,18,19,21,22,23,28,29,151,174]
		self.imageProcessor = ImProc()
		self.homeD = '/home/planktoncam/'
		self.debug = debug

		#check for master list
		self.updated = True
		try:
			self.X_ = np.load("X_.npy")
			self.y_ = np.load("y_.npy")
		except IOError:
			if(self.debug):
				print "Could not find master array"
			self.updated = False
		if(self.updated):
			self.addImgsToMaster()
			if(self.debug):
				print "Adding to Master List"
		else:
			self.genMasterTrainSet()
			if(self.debug):
				print "Generating Master List"

	def genMasterTrainSet(self):
		self.X_ = np.array([],ndim=2)
		self.y_ = np.array([])
		#allTaxa = self.taxaCollection.find({},{"_id":1})
		allTaxa = self.taxaCollection.find({})
		if(self.debug):
			print "Found " + str(len(allTaxa)) + " taxa"
		self.numClasses = len(allTaxa)
		for taxa in allTaxa
			classID = taxa._id
			classJSON = {'userTags.taxaTag':classID}} 
			foundClassifiedImages = self.imgCollection.find(classJSON)
			self.currentTrainLength = len(foundClassifiedImages)
			if(self.debug):
				print "Found " + str(self.currentTrainLength) + " training images for class"
			for classTrainImgs in foundClassifiedImages:
				for img in classTrainImgs:
					self.imgUpdate(img)
		self.storeMatrix()
		self.trainClassifiers()

	def addImgsToMaster(self):
		untrained = self.imgCollection.find({"trained":"0"})
		if(self.debug):
			print "Found " + str(len(untrainedImg)) + " untrained images"
		if (len(untrained) < self.limitToUpdate):
			raise SystemExit
		for untrainedImg in untrained:
			self.imgUpdate(untrainedImg)
		self.storeMatrix()
		self.trainClassifiers()

	def imgUpdate(self,untrainedImg):
		features = self.ImageProcess(untrainedImg.filepath)
		if (len(self.X_) == 0):
			self.X_ = features
			self.y_ = untrainedImg.userTags.taxaTag
		else:
			self.X_ = np.append(self.X_,features,0)
			self.y_ = np.append(self.y_,untrainedImg.userTags.taxaTag,0)
		self.imgCollection.update({"_id": untrainedImg._id},{'$set':{"trained":"1"}})

	def ImageProcess(self,imagePath):
		#send one imgPath to processor and return np.array of features
		path = os.path.join(self.homeD,imagePath)
		return self.imageProcessor.processImg(path)

	def storeMatrix(self):
		np.save("X_.npy",self.X_)
		np.save("y_.npy",self.y_)

	def trainClassifiers(self):
		#create NB classifier based on current data
		if(self.debug):
			print "Training Classifiers on training data"
		self.genNBTrainSet()
		self.nb = NaiveBayes(self.X_train, self.y_train)
		self.nb.train(self.nbSelectedFeatures)

		cls = set(self.y_train)
		self.nb.numClasses = len(cls)
		self.nb.classIDs = cls

		self.storeClassifiers()

	def genNBTrainSet(self):
		#extract relevent NB features from master
		self.X_train = self.X_
		self.y_train = self.y_

	def genSVMTrainSet(self):
		#extract relevent SVM features from master
		self.X_train = self.X_
		self.y_train = self.y_

	def storeClassififers(self):
		if(self.debug):
			print "Pickle-ing Classifiers"

		t = time.time()
		direct = '/home/planktoncam/apps/python/planktoncv/classifiers/'
		self.nb.id = self.nb.name + str(int(t)) 
		filepath = self.nb.id+".p"
		
		pickle.dump(self.nb, open(os.path.join(direct,filepath), "wb"))
		
		clfJSON={
			"name":self.nb.name,
			"time":t,
			"id":self.nb.id
			"trainSize":len(self.self.nb.y_),
			"testSize":0,
			"score":0,
			"classes":self.nb.numClasses,
			"classID":self.nb.classIDs,
			"filepath": filepath
		}

		if(self.classCollection.find({"id":clfJSON['id']}).count() == 0 ):
			self.classCollection.insert(clfJSON)

class TestPhase:

	def __init__(self, imgList, ip='localhost',port='27017',debug=False):
		#classifier variables
		self.classCollectionName = 'classifiers'
		self.imgCollectionName = 'planktonDB'
		self.taxaCollectionName = 'taxaTags'

		#port and ip
		self.ip = ip
		self.port = port

		#connect to db
		client = MongoClient(self.ip,int(self.port))
		self.db = client.planktonDB

		#connect to each important collection
		self.classCollection = self.db[self.classCollectionName]
		self.imgCollection = self.db[self.imgCollectionName]
		self.taxaCollection = self.db[self.taxaCollectionName]

		#other var
		self.imgList = imgList
		self.imageProcessor = ImProc()
		self.homeD = '/home/planktoncam/'
		self.debug=debug

		#run
		self.extractImgs()

	def extractImgs(self):
		self.imgDocs = []
		self.imgsToClassify = []
		self.imgIDs = []
		self.loadClassifiers()
		for imgPath in self.imgList:
			imgdoc=self.imgCollection.find({"filepath":imgPath})

			#**** CHECK IF IMG HAS ALREADY BEEN CLASSIFIED *****
			self.imgDocs.append(imgdoc)
			self.imgsToClassify.append(imgPath)
			self.imgIDs.append(imgdoc.attributes.id)
		self.imageProcess()

	def loadClassifiers(self):
		classifiers = self.classCollection.find().sort({"time":-1})
		self.c1 = classifiers[0]
		self.c2= classifiers[1]
		self.direct = '/home/planktoncam/'
		path1 = os.path.join(self.direct,self.c1.filepath)
		self.classifier1 = pickle.load(open(path1,'rb'))
		path2 = os.path.join(self.direct,self.c2.filepath)
		self.classifier2 = pickle.load(open(path2,'rb'))

	def imageProcess(self):
		self.X_test = np.array([],ndim=2)
		for cnt,imgPath in enumerate(self.imgsToClassify):
			path = os.path.join(self.homeD,imgPath)
			fArray = self.imageProcessor.processImg(path)
			if(cnt == 1):
				self.X_test = fArray
			else:
				self.X_test = np.append(self.X_test,fArray)
		self.classify()

	def classify(self):
		self.y_classes1 = self.c1.classID
		self.y_classes2 = self.c2.classID
		self.X_result1 = self.classifier1.test(self.X_test)
		self.X_result2 = self.classifier2.test(self.X_test)
		for img in imgsToClassify:
			resJSON={
					self.c1.name : self.X_result1[i],
					self.c2.name : self.X_result2[i],
					}
			self.imgCollection.update({"filepath":img},{'$set':{"classififed":resJSON}})
		# self.histData1 = np.append(self.histData1,np.array(self.X_result1))
		# self.histData2 = np.append(self.histData2,np.array(self.X_result2))
		self.storeResults()

	def storeResults(self):
		self.histData1 = np.array([],ndim=2)
		self.histData2 = np.array([],ndim=2)
		name1 = [self.c1.name+"_post.npy", self.c1.name+"_imgID.npy"]
		name2 = [self.c2.name+"_post.npy", self.c2.name+"_imgID.npy"]

		try:
			storeProbArray = np.load(name1[0])
			self.histData1 = np.append(storeProbArray,np.array(self.X_result1))
			#storeImgArray = np.load(name1[1])
		except IOError:
			self.histData1 = np.array(self.X_result1)

		try:
			storeProbArray = np.load(name2[0])
			self.histData2 = np.append(storeProbArray,np.array(self.X_result2))
			#storeArray = np.load(name2)
		except IOError:
			self.histData2 = np.array(self.X_result2)

		np.save(self.histData1, name1[0])
		np.save(self.histData2, name2[0])

class UserPhase:

	def __init__(self,stuff):
		pass