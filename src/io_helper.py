###############################################################################

from pandas import read_csv, get_dummies

from config_helper import ConfigHelper



class IOHelper():

	data_path = "data/"
	results_path = "results/"


	@staticmethod
	def _write_to_csv(frame, filename, filepath, precision):
		file = filepath+filename+".csv"
		frame.to_csv(path_or_buf=file, encoding="ascii",
						 float_format=precision)

	@staticmethod
	def _read_csv(filename, header, index_col, sep, na):

		file = IOHelper.data_path+filename+".csv"
		frame = read_csv(filepath_or_buffer=file, encoding="ascii", 
						 index_col=index_col, header=header,
						 sep=sep, na_values=na)

		frame.reset_index(inplace=True)
		frame.drop(columns=frame.columns[0], inplace=True)

		if na != None:
			frame.dropna(inplace=True)

		return frame


	@staticmethod
	def read_dataset(filename):
		print("Reading dataset: " + filename)

		frame = None
		target = -1
		comma = ","
		space = " "

		if filename=="blood":
			frame = IOHelper._read_csv(filename, 0, None, comma, None)
		elif filename=="breast":
			frame = IOHelper._read_csv(filename, None, 0, comma, "?")
		elif filename=="chess":
			frame = IOHelper._read_csv(filename, None, None, comma, None)
			frame.drop(columns=14, inplace=True)
			frame = get_dummies(frame, drop_first=True)
		elif filename=="german":
			frame = IOHelper._read_csv(filename, None, None, space, None)
			frame.drop(columns=[0,2,3,5,6,8,9,11,13,14,16], inplace=True)
			frame = get_dummies(frame, drop_first=True)
			target = 20
		elif filename=="heart":
			frame = IOHelper._read_csv(filename, None, None, space, None)
		elif filename=="ionosphere":
			frame = IOHelper._read_csv(filename, None, None, comma, None)
		elif filename=="liver":
			frame = IOHelper._read_csv(filename, None, None, comma, None)
			frame.drop(columns=6, inplace=True)
			frame.loc[frame[5] < 3, 5] = 0
			frame.loc[frame[5] >= 3, 5] = 1
		elif filename=="parkinsons":
			frame = IOHelper._read_csv(filename, 0, "name", comma, None)
			target = "status"
		elif filename=="sonar":
			frame = IOHelper._read_csv(filename, None, None, comma, None)
		elif filename=="spambase":
			frame = IOHelper._read_csv(filename, None, None, comma, None)
		else:
			raise ValueError("No existent information for set named "+set_name)

		return frame, target

	@staticmethod
	def store_results(frame, filename):
		print("Storing results")

		IOHelper._write_to_csv(frame, filename, IOHelper.results_path,
							   precision=None)