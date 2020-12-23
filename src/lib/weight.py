
import pandas as pd
import numpy as np

def calc_personlised_weight(ed_matrix, dtopic_matrix, topic_vec, alpha):
	''' This function calculates the personalised weight of documents and experts

	Paramaters
	----------
	ed_matrix: pd.DataFrrame
		An expert-document matrix
	dtopic_matrix: pd.DataFrame
		A document-topic matrix
	topic_vec: np.ndarray
		A 1-dim array containing weights for each topic
	alpha: int or float (alpha in [0, 1])
		An alpha value for smoothing factor
	
	Return
	------
	doc_df: pd.DataFrame
		A personalised document-topic matrix
	expert_df: pd.DataFrame
		A personalised expert-topic matrix
	'''
	# Valiation
	if not isinstance(ed_matrix, pd.DataFrame):
		raise TypeError("`ed_matrix` should be type of pd.DataFrame, but received {}".format(type(ed_matrix)))
	if not isinstance(dtopic_matrix, pd.DataFrame):
		raise TypeError("`dtopic_matrix` should be type of pd.DataFrame, but received {}".format(type(dtopic_matrix)))	
	if not isinstance(topic_vec, np.ndarray):
		raise TypeError("`topic_vec` should be type of np.ndarray, but received {}".format(type(topic_vec)))
	if alpha > 1.0 or alpha < 0.0:
		raise ValueError("`alpha = {}` is out of range between 0 and 1 inclusively.".format(alpha))

	# Initialise
	expert_ids = ed_matrix.index
	doc_ids = dtopic_matrix.index
	topics = dtopic_matrix.columns
	topic_vec = topic_vec.reshape(-1, 1)

	# Calculate personalisation for document
	pr_doc = dtopic_matrix.values.dot(1 - alpha) + topic_vec.dot(alpha)

	# Calculate personalisation for expert
	pr_expert = np.matmul(ed_matrix.values, pr_doc)

	# Construct dataframe
	doc_df = pd.DataFrame(pr_doc, index=doc_ids, columns=topics)
	expert_df = pd.DataFrame(pr_expert, index=expert_ids, columns=topics)

	return expert_df, doc_df
