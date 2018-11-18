###############################################################################

from pandas import read_csv



class IOHelper():
	data_path = "data/"
	results_path = "results/"

	@staticmethod
	def _write_to_csv(dataframe, filename, filepath, precision):
		file = filepath+filename+".csv"
		dataframe.to_csv(path_or_buf=file, encoding="ascii",
						 float_format=precision)

	@staticmethod
	def read_dataset(filename):
		print("Reading dataset")
		
		file = IOHelper.data_path+filename+".csv"
		dataframe = read_csv(filepath_or_buffer=file, encoding="ascii", 
							 skip_blank_lines=False)
		return dataframe

	@staticmethod
	def store_results(dataframe, filename):
		print("Storing results")

		IOHelper._write_to_csv(dataframe, filename, IOHelper.results_path,
							   precision=None)