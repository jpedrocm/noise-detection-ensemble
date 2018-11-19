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
				("FL_RF", Tree(), "fl"),
				("CL_RF", Tree(), "cl"),
				("FL_MAJ_RF", RF(n_estimators=501, n_jobs=-1), "maj"),
				("RF", RF(n_estimators=500, n_jobs=-1), None),
				("Boosting", Adaboost(base_estimator=Tree(max_depth=30, 
									  min_samples_split=20, 
									  min_impurity_decrease=0.01), 
									  n_estimators=100, algorithm="SAMME"),
									  None)
				]