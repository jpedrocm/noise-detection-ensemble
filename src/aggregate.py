###############################################################################

from config_helper import ConfigHelper
from io_helper import IOHelper
from metrics_helper import MetricsHelper



def aggregate():
	
	datasets = ConfigHelper.get_datasets()
	results = IOHelper.read_multiple_results("final_", datasets)
	MetricsHelper.adapt_results(results)

	aggregated = MetricsHelper.aggregate_rate(results)

	aggregated = MetricsHelper.aggregate_threshold(results)

	aggregated = MetricsHelper.aggregate_error(results)
	IOHelper.store_error_table(aggregated, "error_table")

	aggregated = MetricsHelper.aggregate_filter(results)


if __name__=="__main__":
	aggregate()