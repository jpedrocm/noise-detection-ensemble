###############################################################################

from copy import deepcopy

from numpy import inf as INF
from numpy import mean, float64

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame, Series

from data_helper import DataHelper
from metrics_helper import MetricsHelper

EPS=10**-4


class NoiseDetectionEnsemble():

	k_folds = 3
	sampling_rates = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
	clean_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


	@staticmethod
	def get_ensemble(base_clf, use_oob, sample_rate, max_nb_feats):
		return BaggingClassifier(base_clf, n_estimators=501,
					oob_score=use_oob, n_jobs=-1,
					max_samples=min(sample_rate, 1.0),
					max_features=max_nb_feats)

	@staticmethod
	def _first_stage(base_clf, X, y, max_nb_feats):
		min_error = float64(INF)
		ideal_rate = None
		best_ensemble = None

		for rate in NoiseDetectionEnsemble.sampling_rates:

			X_adapted, y_adapted, adapted_rate = DataHelper.adapt_rate(X, y, rate)

			ensemble = NoiseDetectionEnsemble.get_ensemble(base_clf, True,
												adapted_rate, max_nb_feats)

			ensemble.fit(X_adapted, y_adapted)

			error = (1-ensemble.oob_score_)*100

			if error < min_error - EPS:
				min_error = error
				ideal_rate = rate
				best_ensemble = ensemble

		return (best_ensemble, ideal_rate, min_error)

	@staticmethod
	def _get_oob_prediction_matrix(detector, X):

		estimators_samples_set = [set(samples) for samples in \
							detector.estimators_samples_]

		nb_clfs = len(estimators_samples_set)
		nb_instances = len(X)

		ausence_mask = [[iloc_idx not in estimators_samples_set[i] for i in \
						range(nb_clfs)] for iloc_idx in range(nb_instances)]

		instance_ausence = DataFrame(ausence_mask, index=X.index,
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
		oob_preds = [[val for val in row if not isinstance(val, bool)] \
					for row in predictions]

		return oob_preds

	@staticmethod
	def _get_major_oob_label(oob_pred_matrix):

		oob_preds = Series([val for row in oob_pred_matrix for val in row])
		return oob_preds.mode()[0]

	@staticmethod
	def _mark_as_noisy(oob_matrix, y, threshold):

		nb_instances = len(y)

		errors = [MetricsHelper.calculate_error_score([y.iloc[i]]*len(oob_matrix[i]), 
						oob_matrix[i]) for i in range(nb_instances)]

		is_noise_list = [error > threshold  for error in errors]
		is_noise = Series(is_noise_list, index=y.index, dtype=bool, name="is_noise")
		return is_noise

	@staticmethod
	def _clean_noisy_data(train_X, train_y, is_y_noise, clean_type, 
						  maj_oob_label):

		clean_X = None
		clean_y = None

		noise_idxs = is_y_noise[is_y_noise==True].index

		if clean_type=="fl":
			clean_X = train_X.drop(index=noise_idxs)
			clean_y = train_y.drop(index=noise_idxs)

		elif clean_type=="cl":

			clean_X = train_X
			clean_y = deepcopy(train_y)
			clean_y.loc[noise_idxs] = maj_oob_label
		else:
			raise ValueError("Clean type error: " + clean_type)

		return (clean_X, clean_y)

	@staticmethod
	def _calculate_cv_error(base_clf, best_rate, X, y, is_y_noise, clean_type, 
							max_nb_feats, major_oob_label):

		errors = []

		skf = StratifiedKFold(n_splits=NoiseDetectionEnsemble.k_folds, 
								shuffle=True)

		for train_idxs, val_idxs in skf.split(X=range(len(y)), y=y):

			train_X = DataHelper.select_rows(X, train_idxs, copy=False)
			train_y = DataHelper.select_rows(y, train_idxs, copy=False)
			train_is_y_noise = DataHelper.select_rows(is_y_noise, train_idxs,
												copy=False)
	
			clean_train = NoiseDetectionEnsemble._clean_noisy_data(train_X,
													train_y, train_is_y_noise,
													clean_type, major_oob_label)

			train_X, train_y, adapted_rate = DataHelper.adapt_rate(clean_train[0], 
																clean_train[1], 
																best_rate)

			ensemble = RF(501, n_jobs=-1, max_features="sqrt")
			ensemble.fit(train_X, train_y)

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

		oob_pred_matrix = NoiseDetectionEnsemble._get_oob_prediction_matrix(
																detector, X)

		maj_oob_label = NoiseDetectionEnsemble._get_major_oob_label(
															oob_pred_matrix)

		for threshold in NoiseDetectionEnsemble.clean_thresholds:
			is_y_noise = NoiseDetectionEnsemble._mark_as_noisy(oob_pred_matrix,
															   y, threshold)

			cv_error = NoiseDetectionEnsemble._calculate_cv_error(base_clf,
																  best_rate,
																  X, y,
																  is_y_noise,
																  clean_type,
																  max_nb_feats,
																  maj_oob_label)
			
			if cv_error < min_error:
				min_error = cv_error
				best_threshold = threshold
				best_is_y_noise = is_y_noise
		
		cleansed_tuple = NoiseDetectionEnsemble._clean_noisy_data(X, y,
														best_is_y_noise,
														clean_type, 
														maj_oob_label)

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