
import pandas as pd
import numpy as np
import math

def calc_ngram_tfidf(model_dict, method="indirect"):
	''' This function runs nTFIDF calculation tasks

	Parameters
	----------
	model_dict: dict
		A dictionary containing tf infomration for terms and phrases
	method: str
		A calculation method for nIDF ("indirect" - default | "direct")

	Return
	------
	res: np.array
		A nTFIDF matrix of phrases
	sel_phrases: array_like
		A list of selected phrases
	'''
	# Parse model
	tft_model = model_dict['tft_model']
	tft_trans = model_dict['tft_trans']
	tfp_model = model_dict['tfp_model']
	tfp_trans = model_dict['tfp_trans']

	# Preparing data
	terms 			= tft_model.get_feature_names()
	tf_terms 		= tft_trans.todense()
	idx_terms 		= {k: idx for idx, k in enumerate(terms)}
	phrases 		= tfp_model.get_feature_names()
	tf_phrases  	= tfp_trans.todense()
	dfg_phrases 	= np.count_nonzero(np.array(tf_phrases), axis=0)
	idx_phrases 	= {k: idx for idx, k in enumerate(phrases)}
	doc_len 		= tf_terms.shape[0]

	# Intialise for N-gram TFIDF
	shape = (doc_len, len(idx_phrases))
	res = np.zeros(shape).astype(np.float32)
	pos_res = []

	# Calculating N-gram TFIDF
	for ind, phrase in enumerate(phrases):
		# Find candidates
		terms = phrase.split('_')
		candidates = []
		for term in terms:
			if term in idx_terms:
				t_pos = idx_terms[term]
				candidates.append(t_pos)

		# Calculate n-tfidf for existing candidates
		if len(candidates) != 0:
			tf_vec = tf_terms[:, candidates]
			conjunction = np.logical_and.reduce(tf_vec, axis=1)
			df_theta = np.count_nonzero(conjunction)			
			phrase_pos = idx_phrases[phrase]
			dfg = dfg_phrases[phrase_pos]

			# Calculate nIDF
			if method == "indirect":
				idf = math.log(np.divide((doc_len * dfg) + 1, pow(df_theta, 2) + 1)) + 1
			elif method == "direct":
				idf = math.log(np.divide(doc_len + 1, df_theta + 1)) + 1
			
			# Only consider IDF > 0 
			if idf > 0:
				pos_res.append(phrase_pos)

			# Calculate TF
			tf = np.divide(np.sum(tf_vec, axis=1),len(terms))

			# Calculate TFIDF
			tfidf = np.array(np.multiply(tf, idf))[:, 0]
			res[:, phrase_pos] = tfidf
	
	# Only consider IDF > 0 
	res = res[:, pos_res]
	sel_phrases = np.array(phrases)[pos_res]
	return res, sel_phrases		

def calc_pr_weight(ed_matrix, dtopic_matrix, topic_vec, alpha):
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

	# Calculate personalisation for document
	pr_doc = dtopic_matrix.values.dot(1 - alpha) + topic_vec.dot(alpha)

	# Calculate personalisation for expert
	pr_expert = np.matmul(ed_matrix.values, pr_doc)

	# Construct dataframe
	doc_df = pd.DataFrame(pr_doc, index=doc_ids, columns=topics)
	expert_df = pd.DataFrame(pr_expert, index=expert_ids, columns=topics)

	return expert_df, doc_df
