###############################################################################

from copy import deepcopy
from numpy import sqrt
from pandas import Series

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder



class DataHelper():

	label_mapping = None


	@staticmethod
	def extract_feature_labels(frame, target, range_cols=True):

		labels = frame[target] if target!=-1 else frame.iloc[:, target]

		feats = frame.drop(columns=labels.name)

		le = LabelEncoder()
		encoded_labels = Series(le.fit_transform(labels), index=labels.index,
								dtype="category")

		if range_cols:
			feats.columns=range(len(feats.columns))

		return feats, encoded_labels

	@staticmethod
	def split_in_sets(frame, labels):

		skf = StratifiedKFold(n_splits=3, shuffle=True)
		splits = skf.split(X=range(len(labels)), y=labels)
		return list(splits)[0]

	@staticmethod
	def select_rows(frame, idxs, copy):

		sel = frame.iloc[idxs]
		sel = deepcopy(sel) if copy == True else sel
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
		return sample_idxs, DataHelper.map_labels(labels,
								sample_idxs, sample_values)

	@staticmethod
	def calculate_max_nb_features(features):

		nb_features = len(features.columns)
		return max(1, int(sqrt(nb_features)))

	@staticmethod
	def adapt_rate(X, y, rate):

		adapted = None

		if rate > 1:
			new_X = deepcopy(X)
			new_X["label"] = deepcopy(y)
			samp = new_X.sample(frac=rate-1)
			new_X = new_X.append(samp)
			new_y = new_X["label"]
			new_X = new_X.drop(columns="label")

			adapted = [new_X, new_y, rate]
		else:
			adapted = [deepcopy(X), deepcopy(y), rate]

		return adapted