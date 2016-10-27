import sys
import numpy as np
import sklearn 
import math
import matplotlib.axes as ax
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import RFE
from itertools import izip as zip, count
from LookUp import look


class NaiveBayes:
	def __init__(self, classData, classLabels, name):
		self.X_=classData
		self.y_=classLabels
		self.lenData = classData.shape[0]
		#self.y_int=np.int_(classLabels).tolist()
		self.H_=[]
		self.prior=[]
		self.total=len(self.y_)
		self.currentClass=0
		self.currentFeature=0
		self.resultsName=name
		self.sumP = np.array([])
		self.imgCnt = 0
		self.name = "NaiveBayesClassifier"

	def standardize(self):
		self.stdData = []
		self.fAvg = []
		self.fStd = []
		for f in self.features:
			t=self.X_[:,f]
			t_m=np.mean(t)
			t_s=np.std(t)
			t_array=np.divide(np.subtract(t,t_m),t_s).tolist()
			self.stdData.append(t_array)
			self.fAvg.append(t_m)
			self.fStd.append(t_s)

	def train(self, selectedFeatures):
		self.features = selectedFeatures
		# print self.lenData
		self.standardize()
		#extract a array of values for each feature
		#self.numClasses = max(self.y_int)+1
		self.y_set = set(self.y_)
		self.numClasses = len(self.y_set)

		self.prior = []
		for classID in self.y_set:
			self.currentClass = classID

			classHist=[]
			ind = [i for i, j in zip(count(), self.y_) if j == classID]
			self.prior.append(float(len(ind))/self.total)
			# end=ind[-1]+1
			# start=ind[0]
			#start=self.y_int.index(c)qqq
			# print "Starts: " + str(start)
			#end=self.y_int.__len__()-self.y_int[::-1].index(c)
			# print "Ends " + str(end)
			#print ind 
			for j,f in enumerate(selectedFeatures):
				# print "j" +str(j)
				self.currentFeature=f
				cft=[]
				for i in ind:
					cft.append(self.stdData[j][i])
				#cft=self.stdData[j][start:end]
				# print "cft is "
				# print cft
				classFt=self.genHist(cft)
				classHist.append(classFt)
			self.H_.append(classHist)
		#print self.H_
		#self.allGraphs()
		return

	def genHist(self, fArray):

		#print fArray
		maxVal = max(fArray)
		minVal = min(fArray)
		if(maxVal==minVal):
			return [0]
		size = len(fArray)
		binCalc = math.sqrt(size)
		#binCalc = size ** (1/3.0) 
		binWidth = (maxVal-minVal)/binCalc
		numBins = int(math.ceil(binCalc))
		bins = []
		for i in range(numBins+1):		
			bins.append(minVal+i*binWidth)
		#print bins
		counts = [0]*(numBins)
		for f in fArray:
			for i in range(numBins):
				if ((f>=bins[i]) & (f<bins[i+1])):
					counts[i]+=1
		if(maxVal>=bins[-1]):
			counts[-1]+=1
		normCounts = [float(j)/size for j in counts]
		#print counts
		#print normCounts
		barBins=bins[0:-1]
		# barBins.pop()
		#normCounts.append(0)
		hs=[normCounts,bins]
		#self.plotHist(fArray,numBins)
		#self.plotBar(barBins, normCounts, binWidth)
		return hs

	def test(self,imgParams):
		self.results = []
		for i,d in enumerate(imgParams):
			self.imgData=d
			self.currentImage=i
			self.results.append(self.posterior())
		return self.results

	def test_l(self,X_,y_):
		self.correctOne = float(0)
		self.correctTwo = float(0)
		self.confusion = np.zeros((self.numClasses,self.numClasses),dtype=np.int)
		for i in range(len(y_)):
			self.imgData=X_[i]
			self.currentImage=i
			p=self.posterior()
			if(sum(p)<(1*10**-18)):
				self.imgCnt+=1
			self.score(p,y_[i])
		self.correctOne=self.correctOne/len(y_)
		self.correctTwo=self.correctTwo/len(y_)
		print "Correct (1) ---> " +str(self.correctOne)
		print "Correct (2) ---> " +str(self.correctTwo)
		print "Bad Images  ---> " +str(self.imgCnt)
		self.probTotals()
		self.printResults()

	def probTotals(self):
		sAvg=np.mean(self.sumP)
		sMin=np.amin(self.sumP)
		print "Avg sum pro " + str(sAvg)
		print "Min sum pro " + str(sMin)

	def printResults(self):
		colors=['r','g','b','c','m','y']
		colLabels=(look.classes(0),look.classes(1),look.classes(2),look.classes(3),look.classes(4),look.classes(5))
		rowLabels=(look.classes(0),look.classes(1),look.classes(2),look.classes(3),look.classes(4),look.classes(5))
		nrows, ncols = self.numClasses, self.numClasses
		hcell, wcell = 0.7, 1.5
		hpad, wpad = 0, 2    
		fig=plt.figure(figsize=(ncols*wcell+wpad, nrows*hcell+hpad-2))
		t="Summary of Naive Bayes Results: \nPercent Correct (1st) = " + str(self.correctOne) + "\nPercent Correct (1st and 2nd) = " + str(self.correctTwo)
		filename=self.resultsName + "_resultsNB.png"
		plt.title(t)
		plt.xlabel('Computer')
		plt.ylabel('Human')
		ax = fig.add_subplot(111)
		ax.axis('off')
		cellText=[]
		print self.confusion
		for row in self.confusion:
			cellText.append(row)
		#do the table
		table = ax.table(cellText=cellText, rowLabels=rowLabels, colLabels=colLabels, rowColours=colors, loc='upper center', fontsize=25, cellLoc='center')
		#the_table = ax.table(cellText=cellText, rowColours=colors, colColours=colors, loc='center', fontsize=25)
		for cidx in table._cells:
			if(cidx[0] == 0):
				table._cells[cidx].set_facecolor(colors[cidx[1]])

		plt.savefig(filename,bbox_inches='tight')
		plt.close()

	def score(self,p,cClass):
		sorted_p=sorted(p)
		user = int(cClass)
		comp = p.index(sorted_p[-1])
		second = p.index(sorted_p[-2])
		self.confusion[user,comp]+=1
		if(user == comp):
			self.correctOne+=1
		if((user == comp) | (user == second)):
			self.correctTwo+=1

	def posterior(self):
		#print self.imgData
		imgF = [(self.imgData[j]-self.fAvg[i])/self.fStd[i] for i,j in enumerate(self.features)]
		#imgF = self.imgData
		condProb=[]
		#print self.H_[0][0][0]
		for eachC in self.H_:
			j=0
			prob=[]
			for eachF in eachC:
				for i in range(len(eachF[0])):
					if((imgF[j]>=eachF[1][i]) & (imgF[j]<eachF[1][i+1])):
						prob.append(eachF[0][i])
						continue
				if((imgF[j]>=max(eachF[1])) | (imgF[j]<min(eachF[1]))):
					prob.append(.0001)
				j+=1
				#print prob
			condProb.append(prob)
		#print condProb
		classProb=[]
		for eachClassProb in condProb:
			totalprob=1
			for prob in eachClassProb:
				totalprob=prob*totalprob
			classProb.append(totalprob)
		self.printPost(classProb)
		# print "Probabilities by class for Img " + str(self.currentImage)
		# for i in range(len(classProb)):
		# 	print look.classes(i) + " --> " + str(classProb[i])
		self.sumP = np.append(self.sumP, sum(classProb))
		for i in range(len(classProb)):
			classProb[i]=classProb[i]*self.prior[i]
		return classProb

	def fSelect(self, numf):
		selector=SelectKBest(score_func=chi2,k=numf)
		#[p,chi]=chi2(np.absolute(self.X_),self.y_)
		selector.fit(np.absolute(self.X_),self.y_)
		topF=selector.get_support(indices=True)
		return topF.tolist()

	def plotBar(self, bins, percent, w):
		plt.bar(bins, percent, width=w, facecolor='red', alpha=0.75)
		plt.xlabel('Feature Values')
		plt.ylabel('Probability')
		filename = "bar_c"+str(self.currentClass)+"_f"+str(self.currentFeature)+".png"
		title = "Histogram of class:" + str(self.currentClass)+ " PDF of feature: "+str(self.currentFeature)
		plt.title(title)
		plt.grid(True)
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	def plotHist(self, fArray, numBins):
		plt.hist(fArray, bins=numBins, stacked=True, facecolor='blue', alpha=0.75)
		plt.xlabel('Feature Values')
		plt.ylabel('Probability')
		filename = "hist_c"+str(self.currentClass)+"_f"+str(self.currentFeature)+".png"
		title = "Histogram of class:" + str(self.currentClass)+ " PDF of feature: "+str(self.currentFeature)
		plt.title(title)
		plt.grid(True)
		plt.savefig(filename, bbox_inches='tight')
		plt.close()

	def allGraphs(self):
		colors=['r','g','b','c','m','y','c']
		for f in range(len(self.features)):
			ax = plt.subplot(111)
			plt.xlabel('Standardized Feature Values')
			plt.ylabel('Probability')
			filename = self.resultsName+ "_feature_"+str(self.features[f])+"_std.png"
			title = "Histogram of feature: "+ str(look.feat(self.features[f]))
			plt.title(title)
			plt.grid(True)
			for j in range(len(self.H_)):
				prob=self.H_[j][f][0]
				bins=self.H_[j][f][1]
				barBins=bins[0:-1]
				bw=barBins[1]-barBins[0]
				ax.bar(barBins, prob, width=bw, facecolor=colors[j], alpha=0.75, label=look.classes(j))
			ax.legend(loc=0)
			plt.savefig(filename, bbox_inches='tight')
			plt.close()

	def printPost(self,prob):
		colors=['r','g','b','c','m','y','x']
		ax = plt.subplot(111)
		plt.xlabel("Class")
		plt.ylabel("Relative Likelihood")
		plt.title("Posterior Probability")
		plt.tick_params( axis='x', which='both', bottom='off', top='off',labelbottom='off')
		width=5
		for i, p in enumerate(prob):
			ax.bar(i*width, p*100, width=width, facecolor=colors[i], alpha=0.75, label=look.classes(i))
		ax.legend(loc=0)
		plt.savefig(self.resultsName+"Img_" +str(self.currentImage)+ "_Prob.png",bbox_inches='tight')
		plt.close()