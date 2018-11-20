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
	def read_dataset(filename, index_col, header, sep):
		print("Reading dataset: " + filename)
		
		file = IOHelper.data_path+filename+".csv"
		dataframe = read_csv(filepath_or_buffer=file, encoding="ascii", 
							 index_col=index_col, header=header, sep=sep)

		dataframe.reset_index(inplace=True)
		dataframe.drop(columns=dataframe.columns[0], inplace=True)
		
		return dataframe

	@staticmethod
	def store_results(dataframe, filename):
		print("Storing results")

		IOHelper._write_to_csv(dataframe, filename, IOHelper.results_path,
							   precision=None)