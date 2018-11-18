###############################################################################


from config_helper import ConfigHelper
from data_helper import DataHelper
from io_helper import IOHelper


def main():

	for set_name, target_col in ConfigHelper.get_datasets():
		print("Dataset: " + set_name)

		data = IOHelper.read_dataset(set_name)

		for e in range(ConfigHelper.nb_executions):
			print("Execution: " + str(e))

			feats, labels = DataHelper.extract_feature_labels(data,
															  target_col)

			train_idxs, test_idxs = DataHelper.split_in_sets(feats, labels)

			train_X = DataHelper.select_rows(feats, train_idxs, copy=False)
			train_y = DataHelper.select_rows(labels, train_idxs, copy=True)
			test_X = DataHelper.select_rows(feats, test_idxs, copy=False)
			test_y = DataHelper.select_rows(labels, test_idxs, copy=True)

			for noise_level in ConfigHelper.noise_levels:
				print("Noise level: " + str(noise_level))

				noisy_train_y = DataHelper.insert_noise(train_y, noise_level)

				for name, clf, clean_type in ConfigHelper.get_classifiers():
					print("Ensemble: " + name)

					for rate in ConfigHelper.sampling_rates:
						print("Rate: " + str(rate))

								



if __name__ == "__main__":
	main()