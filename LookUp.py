import sys

class look():
	@staticmethod
	def classes(loc):
		classes=["appendicularian", "chaetoceros", "diatom_chain", "copepod", "pheocystis", "helix", "jelly02", "nauplii"]
		return classes[loc]
	@staticmethod
	def feat(loc):
		features=[0] * 202

		features[0] = "FD0"
		features[1] = "FD1"
		features[2] = "FD2"
		features[3] = "FD3"
		features[4] = "FD4"
		features[5] = "FD5"
		features[6] = "FD6"
		features[7] = "FD7"
		features[8] = "FD8"
		features[9] = "FD9"
		features[10] = "FD10"
		features[11] = "FD11"
		features[12] = "FD12"
		features[13] = "FD13"
		features[14] = "FD14"
		features[15] = "FD15"
		features[16] = "FD16"
		features[17] = "aspect"
		features[18] = "area1"
		features[19] = "area2"
		features[20] = "area3"
		features[21] = "area4"
		features[22] = "area5"
		features[23] = "fillArea"
		features[24] = "ecc"
		features[25] = "esd"
		features[26] = "en"
		features[27] = "sol"
		features[28] = "features[ii].moments_hu"
		features[29] = "features[ii].moments_hu"
		features[30] = "features[ii].moments_hu"
		features[31] = "features[ii].moments_hu"
		features[32] = "features[ii].moments_hu"
		features[33] = "features[ii].moments_hu"
		features[34] = "features[ii].moments_hu"
		features[35] = "features[ii].moments_hu"
		features[36] = "histFeatures"
		features[37] = "histFeatures"
		features[38] = "histFeatures"
		features[39] = "histFeatures"
		features[40] = "histFeatures"
		features[41] = "histFeatures"
		features[42] = "histFeatures"
		features[43] = "histFeatures"
		features[44] = "histFeatures"
		features[45] = "histFeatures"
		features[46] = "histFeatures"
		features[47] = "histFeatures"
		features[48] = "histFeatures"
		features[49] = "histFeatures"

		if (loc < 50):
			return features[loc]
		if (loc < 98):
			return "rings"
		if (loc < 148):
			return "wedges"
		if (loc < 196):
			return "GLCM_"+str(loc-172)
		else:
			return str(loc)
			