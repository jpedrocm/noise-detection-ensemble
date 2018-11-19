###############################################################################

from math import inf as INF

from numpy import mean
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame

from data_helper import DataHelper
from metrics_helper import MetricsHelper


class NoiseDetectionEnsemble():
	k_folds = 3
	sampling_rates = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0,]
	clean_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	@staticmethod
	def _first_stage(base_clf, X, y, max_nb_feats):
		min_error = INF
		ideal_rate = None
		best_ensemble = None

		for rate in NoiseDetectionEnsemble.sampling_rates:
			print("Rate: " + str(rate))

			ensemble = BaggingClassifier(base_clf, n_estimators=501,
										 oob_score=True, max_samples=rate,
										 n_jobs=-1, max_features=max_nb_feats)
			ensemble.fit(X, y)
			error = ensemble.oob_score_

			if error < min_error:
				min_error = error
				best_rate = rate
				best_ensemble = ensemble

		return (best_ensemble, best_rate, min_error)

	@staticmethod
	def _mark_as_noisy(detector, threshold, X, y):

		estimators_samples_set = [set(samples) for samples in \
							detector.estimators_samples_]

		nb_clfs = len(estimators_samples_set)
		nb_instances = len(X)

		ausence_mask = [[iloc_idx not in estimators_samples_set[i] for i in \
						range(nb_clfs)] for iloc_idx in range(nb_instances)]

		instance_ausence = DataFrame(ausence_mask, index=y.index,
									 columns=range(nb_clfs), dtype=object)

		for clf_index in instance_ausence.columns:
			clf_instance_ausence = instance_ausence[clf_index]
			oob_idxs = clf_instance_ausence[clf_instance_ausence==True]
			oob_X = X.loc[oob_idxs.index]
			clf_used_feats = detector.estimators_features_[clf_index]
			oob_X_feats = oob_X[clf_used_feats]
			oob_predictions = detector.estimators_[clf_index].predict(oob_X_feats)
			clf_instance_ausence.loc[oob_idxs.index] = oob_predictions

		predictions = instance_ausence.values
		filtered = [[val for val in row if not isinstance(val, bool)] \
					for row in predictions]
		#find each gold label
		errors = [DataHelper.calculate_error_score([gold_label]*len(row), 
									row) for row in filtered]
		#print(errors)
		#sys.exit()

		is_noise_list = [error > threshold for error in errors]
		is_noise = Series(is_noise_list, index=y.index, dtype=bool, name="is_noise")

		return is_noise

	@staticmethod
	def _clean_noisy_data(X, y, is_y_noise, clean_type):

		noise_idxs = is_y_noise[is_y_noise==True].index

		clean_X = None
		clean_y = None

		if clean_type=="fl":
			clean_X = X.drop(index=noise_idxs)
			clean_y = y.drop(index=noise_idxs)

		elif clean_type=="cl":
			clean_X = X
			noise_values = DataHelper.select_rows(y, noise_idxs, copy=False)
			clean_y = DataHelper.map_labels(y, noise_idxs, noise_values)

		else:
			raise ValueError("Clean type error: " + clean_type)

		return (clean_X, clean_y)

	@staticmethod
	def _calculate_cv_error(base_clf, best_rate, X, y, is_y_noise, clean_type, 
							max_nb_feats):

		errors = []

		skf = StratifiedKFold(n_splits=NoiseDetectionEnsemble.k_folds, 
								shuffle=True)

		for train_idxs, val_idxs in skf.split(X=range(len(y)), y=y):
			train_X = DataHelper.select_rows(X, train_idxs, copy=False)
			train_y = DataHelper.select_rows(y, train_idxs, copy=False)
			train_is_y_noise = select_rows(is_y_noise, train_idxs, copy=False)

			clean_train = NoiseDetectionEnsemble._clean_noisy_data(train_X,
													train_y, train_is_y_noise,
													clean_type)

			ensemble = BaggingClassifier(base_clf, n_estimators=501,
										 max_samples=best_rate, n_jobs=-1, 
										 max_features=max_nb_feats)
			ensemble.fit(clean_train[0], clean_train[1])

			val_X = DataHelper.select_rows(X, val_idxs, copy=False)
			val_y = DataHelper.select_rows(y, val_idxs, copy=False)

			predictions = ensemble.predict(val_X)
			error = MetricsHelper.calculate_error_score(val_y, predictions)
			errors.append(error)

		return mean(errors)

	@staticmethod
	def _second_stage(detector, base_clf, best_rate, clean_type, X, y, 
						max_nb_feats):

		min_error = INF
		best_threshold = None
		best_is_y_noise = None

		for threshold in NoiseDetectionEnsemble.clean_thresholds:
			is_y_noise = NoiseDetectionEnsemble._mark_as_noisy(detector,
															   threshold, X, y)
			cv_error = NoiseDetectionEnsemble._calculate_cv_error(base_clf,
																  best_rate,
																  X, y,
																  is_y_noise,
																  clean_type,
																  max_nb_feats)
			if cv_error < min_error:
				min_error = cv_error
				best_threshold = threshold
				best_is_y_noise = is_y_noise
		
		cleansed_tuple = NoiseDetectionEnsemble._clean_noisy_data(X, y,
														best_is_y_noise,
														clean_type)

		return (best_threshold, cleansed_tuple[0], cleansed_tuple[1])
			
	@staticmethod 
	def run(base_clf, clean_type, train_X, train_y, max_nb_feats):
		first_triple = NoiseDetectionEnsemble._first_stage(base_clf, train_X,
														 train_y, max_nb_feats)

		best_detector = first_triple[0]
		best_rate = first_triple[1]

		second_triple = NoiseDetectionEnsemble._second_stage(best_detector,
															 base_clf,
															 best_rate,
															 clean_type, 
															 train_X, 
															 train_y,
															 max_nb_feats)

		return [best_rate, second_triple[0], second_triple[1],
					second_triple[2]]