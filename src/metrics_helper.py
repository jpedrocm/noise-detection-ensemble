###############################################################################

from sklearn.metrics import accuracy_score

from pandas import DataFrame



class MetricsHelper():

	metrics = []


	@staticmethod
	def calculate_error_score(true_y, pred_y):
		return (1-accuracy_score(true_y, pred_y))

	@staticmethod
	def calculate_true_filter(y_idxs, noisy_idxs):
		set_idxs = set(list(y_idxs))
		true_filtered = [i for i in list(noisy_idxs) if i not in set_idxs]
		return len(set(true_filtered))

	@staticmethod
	def reset_metrics():
		MetricsHelper.metrics = []

	@staticmethod
	def convert_metrics_to_frame():
		columns = ["dataset", "execution", "noise", "clf", "sampling_rate",
				   "threshold", "test_error", "all_filtered", "true_filtered"]
		return DataFrame(MetricsHelper.metrics, columns=columns)