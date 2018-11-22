###############################################################################

from numpy import nan

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as Adaboost
from sklearn.tree import DecisionTreeClassifier as Tree

from data_helper import DataHelper
from metrics_helper import MetricsHelper
from noise_detection_ensemble import NoiseDetectionEnsemble
from majority_filtering import MajorityFiltering



class ConfigHelper():

	nb_executions = 50
	noise_levels = [0, 0.1, 0.2, 0.3, 0.4]

	metrics_file = "metrics"


	@staticmethod
	def get_datasets():
		space = " "
		comma = ","
		
		return	["blood", 
				 "breast",
				 "chess",
				 "german",
				 "heart",
				 "ionosphere",
				 "liver",
				 "parkinsons",
				 "sonar",
				 "spambase", 
				]

	@staticmethod
	def get_classifiers():
		return 	[
				("FL_RF", Tree(max_depth=None, min_samples_leaf=1), "fl"),
				("CL_RF", Tree(max_depth=None, min_samples_leaf=1), "cl"),
				("FL_MAJ_RF", MajorityFiltering.get_ensemble(), "maj"),
				("RF", RF(n_estimators=501, max_depth=None, 
						  max_features="sqrt", min_samples_leaf=1, 
						  n_jobs=-1), None),
				("Boosting", Adaboost(base_estimator=Tree(max_depth=None, #default 30
									  min_samples_leaf=1, #default 7
									  min_samples_split=2, #default 20
									  min_impurity_decrease=0.01), 
									  n_estimators=501, algorithm="SAMME"),#defaukt 100
									  None)
				]

	@staticmethod
	def choose_algorithm(clf, clean_type, train_X, noisy_train_y,
						noisy_idxs, max_nb_feats):
		chosen_rate = nan
		chosen_threshold = nan
		chosen_X = None
		chosen_y = None
		chosen_clf = None
		true_filtered = 0

		if clean_type == None:
			chosen_X = train_X
			chosen_y = noisy_train_y
			chosen_clf = clf

		elif clean_type == "maj":
			filt_X, filt_y = MajorityFiltering.run(train_X, 
												   noisy_train_y)
			chosen_X = filt_X
			chosen_y = filt_y
			chosen_clf = MajorityFiltering.get_ensemble()
			true_filtered = MetricsHelper.calculate_true_filter(chosen_y.index,
															noisy_idxs)
		else:
			algorithm_data = NoiseDetectionEnsemble.run(clf, clean_type,
										   		train_X,
								   		   		noisy_train_y, 
								   		   		max_nb_feats)
			chosen_rate = algorithm_data[0]
			chosen_threshold = algorithm_data[1]
			chosen_X = algorithm_data[2]
			chosen_y = algorithm_data[3]
			chosen_X, chosen_y, adapted_rate = DataHelper.adapt_rate(chosen_X,
														chosen_y, chosen_rate)

			chosen_clf = NoiseDetectionEnsemble.get_ensemble(clf, False,
															adapted_rate,
															max_nb_feats)

			true_filtered = MetricsHelper.calculate_true_filter(chosen_y.index,
															noisy_idxs)

		tot_filtered = len(train_X)-len(chosen_X.index.unique())

		return [chosen_rate, chosen_threshold, chosen_X, chosen_y, chosen_clf,
				tot_filtered, true_filtered]