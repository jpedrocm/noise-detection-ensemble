###############################################################################

from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit



class DataHelper():

	@staticmethod
	def extract_feature_labels(dataframe, target_col):
		print "Extracting features and labels"

		labels = dataframe[target_col]
		feats = dataframe.drop(columns=target_col)

		print "Columns: " + str(len(feats.columns))
		print "Rows: " + str(len(labels))
		return feats, labels

	@staticmethod
	def split_in_sets(dataframe, labels):

		skf = StratifiedKFold(n_splits=3, shuffle=True)
		return skf.split(X=range(len(labels)), y=labels)[0]

	@staticmethod
	def select_rows(dataframe, idxs, copy):
		print "Selecting rows of data"

		sel = dataframe.iloc[idxs]
		sel = deepcopy(sel) if copy == True else sel

		print "Rows: " + str(len(sel))
		return sel

	@staticmethod
	def insert_noise(dataframe, level):
		pass
