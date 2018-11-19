###############################################################################


from config_helper import ConfigHelper
from io_helper import IOHelper
from data_helper import DataHelper
from noise_detection_ensemble import NoiseDetectionEnsemble


def main():

	for set_data in ConfigHelper.get_datasets():
		set_name = set_data[0]
		set_index = set_data[1]
		set_target = set_data[2]
		set_header = set_data[3]
		set_sep = set_data[4]

		data = IOHelper.read_dataset(set_name, set_index, set_header, set_sep)

		feats, labels = DataHelper.extract_feature_labels(data, set_target)
		DataHelper.create_label_mapping(labels)
		max_nb_feats = DataHelper.calculate_max_nb_features(feats)

		for e in range(ConfigHelper.nb_executions):
			print("Execution: " + str(e))

			train_idxs, test_idxs = DataHelper.split_in_sets(feats, labels)

			train_X = DataHelper.select_rows(feats, train_idxs, copy=False)
			train_y = DataHelper.select_rows(labels, train_idxs, copy=False)
			test_X = DataHelper.select_rows(feats, test_idxs, copy=False)
			test_y = DataHelper.select_rows(labels, test_idxs, copy=False)

			for noise_level in ConfigHelper.noise_levels:
				print("Noise level: " + str(noise_level))

				noisy_train_y = DataHelper.insert_noise(train_y, noise_level)

				for name, clf, clean_type in ConfigHelper.get_classifiers():
					print("Ensemble: " + name)

					NoiseDetectionEnsemble.run(clf, clean_type, train_X,
											   noisy_train_y, max_nb_feats)


								



if __name__ == "__main__":
	main()