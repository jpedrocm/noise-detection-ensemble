###############################################################################

from pandas import read_csv



class IOHelper():

	data_path = "data/"
	results_path = "results/"


	@staticmethod
	def _write_to_csv(frame, filename, filepath, precision):
		file = filepath+filename+".csv"
		frame.to_csv(path_or_buf=file, encoding="ascii",
						 float_format=precision)

	@staticmethod
	def read_dataset(filename, index_col, header, sep):
		print("Reading dataset: " + filename)
		
		file = IOHelper.data_path+filename+".csv"
		frame = read_csv(filepath_or_buffer=file, encoding="ascii", 
						 index_col=index_col, header=header, sep=sep)

		frame.reset_index(inplace=True)
		frame.drop(columns=frame.columns[0], inplace=True)
		
		return frame

	@staticmethod
	def store_results(frame, filename):
		print("Storing results")

		IOHelper._write_to_csv(frame, filename, IOHelper.results_path,
							   precision=None)