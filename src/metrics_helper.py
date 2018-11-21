###############################################################################

from sklearn.metrics import accuracy_score

from pandas import DataFrame



class MetricsHelper():

	metrics = []


	@staticmethod
	def calculate_error_score(true_y, pred_y):
		return (1-accuracy_score(true_y, pred_y))

	@staticmethod
	def reset_metrics():
		MetricsHelper.metrics = []

	@staticmethod
	def convert_metrics_to_frame():
		columns = ["dataset", "execution", "noise", "clf", "sampling_rate",
				   "threshold", "test_error"]
		return DataFrame(MetricsHelper.metrics, columns=columns)