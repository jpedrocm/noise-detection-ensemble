###############################################################################


from config_helper import ConfigHelper
from data_helper import DataHelper
from io_helper import IOHelper


def main():

	for set_name in ConfigHelper.datasets:
		print("Dataset: " + set_name)

		data = IOHelper.read_dataset(set_name)

		for e in range(ConfigHelper.nb_executions):
			print("Execution: " + str(e))

			train_data, test_data = DataHelper.split_in_sets(data)
			train_X, train_y = DataHelper.extract_labels(train_data)
			test_X, test_y = DataHelper.extract_labels(test_data)

			for noise_level in ConfigHelper.noise_levels:
				print("Noise level: " + str(noise_level))

				noisy_train_y = DataHelper.insert_noise(train_y, noise_level)
				noisy_test_y = DataHelper.insert_noise(test_y, noise_level)

				for name, clf, clean_type in ConfigHelper.get_classifiers():
					print("Ensemble: " + name)

					for rate in ConfigHelper.sampling_rates:
						print("Rate: " + str(rate))

								



if __name__ == "__main__":
	main()