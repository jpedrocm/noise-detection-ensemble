###############################################################################

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier as Adaboost
from sklearn.tree import DecisionTreeClassifier as Tree


class ConfigHelper():
	nb_executions = 50
	noise_levels = [0, 0.1, 0.2, 0.3, 0.4]

	metrics_file = "metrics"


	@staticmethod
	def get_datasets():
		return	[("blood",		 None,		-1,  	0,	","), 
				 ("breast",		 	0,		-1,  None,	","),
				 ("chess",  	 None,		-1,  None,	","), 
				 ("german", 	 None, 		-1,	 None,	" "),
				 ("heart",  	 None, 		-1,	 None,	" "), 
				 ("ionosphere",  None, 		-1,	 None,	","),
				 ("parkinsons",  None, "status",	0,	","), 
				 ("spambase", 	 None, 		-1,	 None,	","), 
				 ("tic-tac-toe", None, 		-1,	 None,	",")
				]

	@staticmethod
	def get_classifiers():
		return 	[
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