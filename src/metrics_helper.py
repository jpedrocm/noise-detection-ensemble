###############################################################################

from sklearn.metrics import accuracy_score

from pandas import DataFrame

import matplotlib.pyplot as plt



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
				   "threshold", "test_error", "true_filtered", "false_filtered"]
		return DataFrame(MetricsHelper.metrics, columns=columns)

	def adapt_results(results):

		results.drop(columns="execution", inplace=True)
		results[["noise", "test_error", "false_filtered", "true_filtered"]] *= 100
		results[["noise"]] = results[["noise"]].astype(int)

	@staticmethod
	def aggregate_rate(results):

		frame = results.drop(columns=["false_filtered", "true_filtered", 
									"threshold", "test_error"])
		frame = frame[frame["clf"]=="FL_RF"]
		frame = frame.groupby(by=["dataset", "noise"]).mean().unstack()

		p=frame.plot(kind="bar", y="sampling_rate", ylim=(0.0, 1.2))
		p.set_xlabel("Fig. 1 - Best Sampling Rates")
		p.set_ylabel("sampling rate")
		p.xaxis.set_label_coords(0.5, -0.1)
		p.yaxis.set_label_coords(-0.05, 0.5)
		p.legend(loc="center", ncol=5, title="noise", fontsize="medium", 
				labels=["0%", "10%", "20%", "30%", "40%"],
				frameon=False, bbox_to_anchor=(0.5,1.05))
		plt.show()

	@staticmethod
	def aggregate_threshold(results):

		frame = results.drop(columns=["false_filtered", "true_filtered", 
									"sampling_rate", "test_error"])
		frame = frame[frame["clf"]=="FL_RF"]
		frame = frame.groupby(by=["dataset", "noise"]).mean().unstack()

		p=frame.plot(kind="bar", y="threshold", ylim=(0.5,1.0))
		p.set_xlabel("Fig. 2 - Best Thresholds")
		p.set_ylabel("threshold")
		p.xaxis.set_label_coords(0.5, -0.1)
		p.yaxis.set_label_coords(-0.05, 0.5)
		p.legend(loc="center", ncol=5, title="noise", fontsize="medium", 
			labels=["0%", "10%", "20%", "30%", "40%"],
			frameon=False, bbox_to_anchor=(0.5,1.05))
		plt.show()

	@staticmethod
	def aggregate_error(results):

		frame = results.drop(columns=["false_filtered", "true_filtered", 
									"sampling_rate", "threshold"])
		frame = frame.replace({"FL_RF": "1_FL_RF", "CL_RF": "2_CL_RF",
					"FL_MAJ_RF": "3_FL_MAJ"})

		mean_group = frame.groupby(by=["dataset", "noise", "clf"])
		mean_frame = mean_group.mean().round(1).unstack().astype(str)
		std_frame = mean_group.std().round(1).unstack().astype(str)
		final_frame = mean_frame + " ± " + std_frame
		return final_frame

	@staticmethod
	def aggregate_filter(results):
		frame = results.drop(columns=["test_error", "sampling_rate", "threshold"])
		for noise in [0, 10, 30]:
			noise_frame= frame[frame["noise"]==noise]
			noise_frame = noise_frame.drop(columns="noise")
			for clf in ["FL_RF", "FL_MAJ_RF"]:
				clf_frame= noise_frame[noise_frame["clf"]==clf]
				clf_frame = clf_frame.drop(columns="clf")
				clf_frame = clf_frame.groupby(by=["dataset"]).mean()
				p=clf_frame.plot(kind="bar", stacked=True, title=str(noise)+ "% noise",
								 ylim=(0,60))
				p.set_xlabel(clf)
				p.set_ylabel("percentage")
				p.xaxis.set_label_coords(0.5, -0.1)
				p.yaxis.set_label_coords(-0.05, 0.5)
				p.legend(loc="upper right", ncol=1, labelspacing=-2,
							 title="", fontsize="medium",frameon=False,
							 bbox_to_anchor=(0.98, 0.95), 
							 labels=["correctly filtered", "all filtered"])
				plt.show()