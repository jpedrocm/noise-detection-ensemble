###############################################################################

from copy import deepcopy

from sklearn.model_selection import StratifiedKFold



class DataHelper():

	@staticmethod
	def extract_feature_labels(frame, target):
		print("Extracting features and labels")

		labels = frame[target] if target!=-1 else frame.iloc[:, target]
		feats = frame.drop(columns=labels.name)

		print("Columns: " + str(len(feats.columns)))
		print("Rows: " + str(len(labels)))
		return feats, labels

	@staticmethod
	def split_in_sets(frame, labels):

		skf = StratifiedKFold(n_splits=3, shuffle=True)
		splits = skf.split(X=range(len(labels)), y=labels)
		return list(splits)[0]

	@staticmethod
	def select_rows(frame, idxs, copy):
		print("Selecting rows of data")

		sel = frame.iloc[idxs]
		sel = deepcopy(sel) if copy == True else sel

		print("Rows: " + str(len(sel)))
		return sel

	@staticmethod
	def insert_noise(frame, level):
		pass
