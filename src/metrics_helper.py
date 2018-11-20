###############################################################################

from sklearn.metrics import accuracy_score



class MetricsHelper():

	metrics = []


	@staticmethod
	def calculate_error_score(true_y, pred_y):
		return (1-accuracy_score(true_y, pred_y))