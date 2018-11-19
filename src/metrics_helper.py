###############################################################################

from sklearn.metrics import accuracy_score



class MetricsHelper():

	@staticmethod
	def calculate_error_score(true_y, pred_y):
		return (1-accuracy_score(true_y, pred_y))