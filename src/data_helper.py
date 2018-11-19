###############################################################################

from copy import deepcopy
from numpy import sqrt

from sklearn.model_selection import StratifiedKFold



class DataHelper():

	label_mapping = None


	@staticmethod
	def extract_feature_labels(frame, target):
		print("Extracting features and labels")

		labels = frame[target] if target!=-1 else frame.iloc[:, target]
		labels = labels.astype("category")
		feats = frame.drop(columns=labels.name)
		feats.columns=range(len(feats.columns))

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
	def create_label_mapping(labels):
		unique_values = labels.unique().tolist()
		DataHelper.label_mapping = {unique_values[0]: unique_values[1],
						  			unique_values[1]: unique_values[0]}

	@staticmethod
	def map_labels(labels, sample_idxs, sample_values):
		noise_values = [DataHelper.label_mapping[v] for v in sample_values]
		noisy_labels = deepcopy(labels)
		noisy_labels.loc[sample_idxs] = noise_values
		return noisy_labels

	@staticmethod
	def insert_noise(labels, level):
		sample = labels.sample(frac=level)
		sample_idxs = sample.index
		sample_values = sample.values

		return DataHelper.map_labels(labels, sample_idxs, sample_values)

	@staticmethod
	def calculate_max_nb_features(features):

		nb_features = len(features.columns)
		return max(1, int(sqrt(nb_features)))