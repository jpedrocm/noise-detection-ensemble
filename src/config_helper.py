from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as Adaboost
from sklearn.tree import DecisionTreeClassifier as Tree


class ConfigHelper():
	k_folds = 3
	nb_executions = 50
	noise_levels = [0, 0.1, 0.2, 0.3, 0.4]
	sampling_rates = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
	clean_threshold = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

	metrics_file = "rf_min_samples"
	

	@staticmethod
	def get_classifiers():
		return [
				("Fl_rf", RF(n_estimators=501, oob_score=True, n_jobs=-1), 
							 "fl"),
				("Cl_rf", RF(n_estimators=501, oob_score=True, n_jobs=-1), 
							 "cl"),
				("Fl_maj_rf", RF(n_estimators=501, oob_score=True, n_jobs=-1),
							 "maj"),
				("RF", RF(n_estimators=500, n_jobs=-1), None),
				("Boosting", Adaboost(base_estimator=Tree(max_depth=30, 
									  min_samples_split=20, 
									  min_impurity_decrease=0.01), 
									  n_estimators=100, algorithm="SAMME"),
									  None)
			   ]