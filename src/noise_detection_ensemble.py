###############################################################################

from math import inf as INF

from numpy import mean


class NoiseDetectionEnsemble():
	k_folds = 3
	sampling_rates = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
	clean_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	@staticmethod
	def _first_stage(base_clf, train_X, train_y, max_nb_feats):
		min_error = INF
		ideal_rate = None
		best_ensemble = None

		for rate in NoiseDetectionEnsemble.sampling_rates:
			print("Rate: " + str(rate))

			ensemble = BaggingClassifier(base_clf(), n_estimators=501,
										bootstrap=True, max_samples=rate, 
										max_features=max_nb_feats,
										oob_score=True, n_jobs=-1)
			ensemble.fit(train_X, train_y)
			error = ensemble.oob_score_

			if error < min_error:
				min_error = error
				best_rate = rate
				best_ensemble = ensemble

		return (best_ensemble, best_rate)

	@staticmethod
	def _mark_as_noisy(detector, threshold, X, y):
		pass

	@staticmethod
	def _calculate_cv_error(base_clf, X, y, is_y_noisy, clean_type):
		errors = []
		#in each fold of stratified 3cv
				#clean the training instances marked as noisy from train_X
				#makes the base_learner ensemble fits the cleansed data
				#store the generalization error on the remaining fold

		return mean(errors)

	@staticmethod
	def _clean_noisy_data(detector, threshold, X, y, is_y_noisy, clean_type):
		pass

	@staticmethod
	def _second_stage(detector, base_learner, clean_type, X, y):

		min_error = INF
		best_threshold = None
		best_is_y_noisy = None

		for threshold in NoiseDetectionEnsemble.clean_thresholds:
			is_y_noisy = NoiseDetectionEnsemble._mark_as_noisy(detector,
															   threshold, X, y)
			cv_error = NoiseDetectionEnsemble._calculate_cv_error(base_clf, 
																  X, y,
																  is_y_noisy,
																  clean_type)
			if cv_error < min_error:
				min_error = cv_error
				best_threshold = threshold
				best_is_y_noisy = is_y_noisy

		cleansed_tuple = NoiseDetectionEnsemble._clean_noisy_data(detector,
														best_threshold, X, y,
														best_is_y_noisy,
														clean_type)

		return (best_threshold, cleansed_tuple[0], cleansed_tuple[1])
			
	@staticmethod 
	def run(base_clf, clean_type, train_X, train_y, max_nb_feats):
		best_tuple = NoiseDetectionEnsemble._first_stage(base_clf, train_y,
														 train_y, max_nb_feats)

		cleansed_tuple = NoiseDetectionEnsemble._second_stage(best_tuple[0],
															  base_clf,
															  clean_type, 
															  train_y, 
															  train_y)

		return (best_tuple[1], best_threshold, cleansed_tuple[0], 
				cleansed_tuple[1])
