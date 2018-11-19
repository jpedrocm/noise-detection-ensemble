###############################################################################

from math import inf as INF

from numpy import mean
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame
from DataHelper import select_rows


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

		#filtra falsos
		#calcula erro
		#assinala como noise
		oob_predictions_per_estimator = [[for ] for index, row in instance_ausence.iterrows()]

		predictions 

		is_noise = Series(index=y.index, dtype=bool, name="is_noise")

		return is_noise

	@staticmethod
	def _clean_noisy_data(X, y, is_y_noise, clean_type):
		pass

	@staticmethod
	def _calculate_cv_error(base_clf, best_rate, X, y, is_y_noise, clean_type, 
							max_nb_feats):

		errors = []

		skf = StratifiedKFold(n_splits=3, shuffle=True)
		splits = skf.split(X=range(len(y)), y=y)

		for train_idxs, val_idxs in splits:
			train_X = select_rows(X, train_idxs, copy=False)
			train_y = select_rows(y, train_idxs, copy=False)
			train_is_y_noise = select_rows(is_y_noise, train_idxs, copy=False)

			clean_train = NoiseDetectionEnsemble._clean_noisy_data(train_X,
													train_y, train_is_y_noise,
													clean_type)

			ensemble = BaggingClassifier(base_clf, n_estimators=501,
										 oob_score=True, max_samples=best_rate,
										 n_jobs=-1, max_features=max_nb_feats)
			ensemble.fit(clean_train[0], clean_train[1])

			val_X = select_rows(X, val_idxs, copy=False)
			val_y = select_rows(y, val_idxs, copy=False)

			#store the generalization error on the remaining fold

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
		#mark as noisy
		cleansed_tuple = NoiseDetectionEnsemble._clean_noisy_data(X, y,
														best_is_y_noise,
														clean_type)

		return (best_threshold, cleansed_tuple[0], cleansed_tuple[1])
			
	@staticmethod 
	def run(base_clf, clean_type, train_X, train_y, max_nb_feats):
		best_triple = NoiseDetectionEnsemble._first_stage(base_clf, train_X,
														 train_y, max_nb_feats)

		best_detector = best_triple[0]
		best_rate = best_triple[1]

		cleansed_data = NoiseDetectionEnsemble._second_stage(best_detector,
															 base_clf,
															 best_rate,
															 clean_type, 
															 train_X, 
															 train_y,
															 max_nb_feats)

		return (best_rate, best_threshold, cleansed_data[0], cleansed_data[1])