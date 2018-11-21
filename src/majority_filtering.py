###############################################################################

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold

from pandas import DataFrame, Series

from data_helper import DataHelper



class MajorityFiltering():

	k_folds = 3


	@staticmethod
	def get_ensemble():
		return RF(n_estimators=501, max_depth=None, max_features="sqrt",
					min_samples_leaf=1, n_jobs=-1)

	@staticmethod
	def _clean_data(X, y):

		clean_X = DataFrame(columns=X.columns)
		clean_y = Series(name=y.name)


		skf = StratifiedKFold(n_splits=MajorityFiltering.k_folds, 
								shuffle=True)

		for train_idxs, val_idxs in skf.split(X=range(len(y)), y=y):

			train_X = DataHelper.select_rows(X, train_idxs, copy=False)
			train_y = DataHelper.select_rows(y, train_idxs, copy=False)

			ensemble = MajorityFiltering.get_ensemble()
			ensemble.fit(train_X, train_y)

			val_X = DataHelper.select_rows(X, val_idxs, copy=False)

			predictions = ensemble.predict(val_X)

			maintain_idxs = [val_idxs[i] for i in range(len(val_idxs)) \
							if predictions[i]==y.iloc[val_idxs[i]]]

			maintain_X = DataHelper.select_rows(X, maintain_idxs, 
												copy=True)
			maintain_y = DataHelper.select_rows(y, maintain_idxs, 
												copy=True)

			clean_X = clean_X.append(maintain_X, verify_integrity=True, 
									sort=False)
			clean_y = clean_y.append(maintain_y, verify_integrity=True)

		return clean_X, clean_y

	@staticmethod
	def run(train_X, train_y):
		return MajorityFiltering._clean_data(train_X, train_y)