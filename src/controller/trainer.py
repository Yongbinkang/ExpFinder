
from src.algo import expfinder
import pandas as pd

def run_expfinder(topics, params):
	''' This function aims to run the ExpFinder given a list of topics and
	valid parameters
	
	Parameters
	----------
	topics: list(str)
		A list of topics
	params: dict
		A dictionary of parameters for the ExpFinder algorithm (See more at src.algo.expfinder)

	Return
	------
	etop_matrix: pd.DataFrame
		An Expert-Topic dataframe where rows are indexes of experts and columns are indexes of topics
	'''
	# Validate params
	if 'ed_graph' not in params.keys():
		raise ValueError("params is missing the `ed_graph` key")
	if 'ed_matrix' not in params.keys():
		raise ValueError("params is missing the `ed_matrix` key")
	
	# Intialise
	ed_graph = params['ed_graph']
	ed_matrix = params['ed_matrix']

	# Build model
	ExpFinder = expfinder.EF(**params)

	# Transform for each topic
	temp = []
	for topic in topics:
		hub, authority = ExpFinder.transform(topic)
		temp.append(authority.reshape(1, -1)[0])

	etop_matrix = pd.DataFrame(temp, columns=ed_graph.nodes(), index=topics)
	etop_matrix = etop_matrix[ed_matrix.index]

	return etop_matrix