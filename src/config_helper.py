###############################################################################

from numpy import nan

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as Adaboost
from sklearn.tree import DecisionTreeClassifier as Tree

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
		
		return	[("blood",		 None,		-1,  	0,	comma), 
				 ("breast",		 	0,		-1,  None,	comma),
				 ("chess",  	 None,		-1,  None,	comma), 
				 ("german", 	 None, 		-1,	 None,	space),
				 ("heart",  	 None, 		-1,	 None,	space), 
				 ("ionosphere",  None, 		-1,	 None,	comma),
				 ("parkinsons",  None, "status",	0,	comma), 
				 ("spambase", 	 None, 		-1,	 None,	comma), 
				 ("tic-tac-toe", None, 		-1,	 None,	comma)
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
						max_nb_feats):
		chosen_rate = nan
		chosen_threshold = nan
		chosen_X = None
		chosen_y = None
		chosen_clf = None

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
		else:
			algorithm_data = NoiseDetectionEnsemble.run(clf, clean_type,
										   		train_X,
								   		   		noisy_train_y, 
								   		   		max_nb_feats)
			chosen_rate = algorithm_data[0]
			chosen_threshold = algorithm_data[1]
			chosen_X = algorithm_data[2]
			chosen_y = algorithm_data[3]
			chosen_clf = NoiseDetectionEnsemble.get_ensemble(clf, False,
															chosen_rate,
															max_nb_feats)

		return [chosen_rate, chosen_threshold, chosen_X, chosen_y, chosen_clf]