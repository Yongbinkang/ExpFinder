
import numpy as np
from src.lib.tokenization import Tokenizer

def tokenise_doc(corpus, stopwords, max_phrase_len, pattern=None):
	''' This function extracts noun phrases as well as tokens from a document
	in a given corpus

	Parameters
	----------
	corpus: list(str)
		A list of documents
	stopwrods: set(str)
		A set of stopwords
	max_phrase_len: int
		An inclusive maximum lengh of a particular phrase
	pattern: Regex str (default: None)
		A linguistic pattern for extracting phrases		
	
	Return
	------
	res: dict
		A dictionary containing tokens and np
	'''
	# Intialise
	res = {
		'tokens': [],
		'np': []
	}
	tokenizer = Tokenizer(stopwords, grammar=pattern)

	# Process per document in the corpus
	for doc in corpus:
		_, data = tokenizer.transform(doc, max_phrase_len)
		res['tokens'].append(data['tokens'])
		res['np'].append(data['np'])

	return res

# def calc_ngram_tfidf(data_dict, start, end, method="indirect"):
# 	''' This function calculates the nTFIDF value of phrases based on the
# 	given TF information of terms and phrases. Note that `start` and `end`
# 	parameters are used to parallelise the calculation for chunk of phrases.

# 	Parameters
# 	----------
# 	data_dict: dictionary
# 		A dictionary containing information about terms and phrases for the calculation
# 	start: int
# 		A starting index of the chunk
# 	end: int
# 		An ending index of the chunk
# 	method: str
# 		A calculation method for nIDF ("indirect" - default | "direct")

# 	Return
# 	------
# 	res: np.array
# 		A nTFIDF matrix of phrases
# 	pos_res: array_like
# 		A position of selected phrases
# 	'''
# 	# Parse data
# 	doc_len 		= data_dict['doc_len']
# 	phrases 		= data_dict['phrases']
# 	dfg_phrases 	= data_dict['dfg_phrases']
# 	idx_phrases 	= data_dict['idx_phrases']
# 	idx_terms 		= data_dict['idx_terms']
# 	tf_terms 		= data_dict['tf_terms']

# 	# Initialise
# 	shape = (doc_len, len(idx_phrases))
# 	res = np.zeros(shape).astype(np.float32)
# 	pos_res = []

# 	for ind, phrase in enumerate(phrases[start:end:]):
# 		# Find candidates
# 		terms = phrase.split('_')
# 		candidates = []
# 		for term in terms:
# 			if term in idx_terms:
# 				t_pos = idx_terms[term]
# 				candidates.append(t_pos)

# 		# Calculate n-tfidf for existing candidates
# 		if len(candidates) != 0:
# 			tf_vec = tf_terms[:, candidates]
# 			conjunction = np.logical_and.reduce(tf_vec, axis=1)
# 			df_theta = np.count_nonzero(conjunction)			
# 			phrase_pos = idx_phrases[phrase]
# 			dfg = dfg_phrases[phrase_pos]

# 			# Calculate nIDF
# 			if method == "indirect":
# 				idf = math.log(np.divide((doc_len * dfg) + 1, pow(df_theta, 2) + 1)) + 1
# 			elif method == "direct":
# 				idf = math.log(np.divide(doc_len + 1, df_theta + 1)) + 1
			
# 			# Only consider IDF > 0 
# 			if idf > 0:
# 				pos_res.append(phrase_pos)

# 			# Calculate TF
# 			tf = np.divide(np.sum(tf_vec, axis=1),len(terms))

# 			# Calculate TFIDF
# 			tfidf = np.array(np.multiply(tf, idf))[:, 0]
# 			res[:, phrase_pos] = tfidf

# 	# Only consider IDF > 0 
# 	res = res[:, pos_res]
# 	return res, pos_res

# def run_ngram_tasks(model_dict, method="indirect"):
# 	''' This function runs nTFIDF calculation tasks

# 	Parameters
# 	----------
# 	model_dict: dict
# 		A dictionary containing tf infomration for terms and phrases
# 	method: str
# 		A calculation method for nIDF ("indirect" - default | "direct")

# 	Return
# 	------
# 	res: np.array
# 		A nTFIDF matrix of phrases
# 	sel_phrases: array_like
# 		A list of selected phrases
# 	'''


# 	# Preparing data
# 	terms 			= tft_model.get_feature_names()
# 	tf_terms 		= tft_trans.todense()
# 	idx_terms 		= {k: idx for idx, k in enumerate(terms)}
# 	phrases 		= tfp_model.get_feature_names()
# 	tf_phrases  	= tfp_trans.todense()
# 	dfg_phrases 	= np.count_nonzero(np.array(tf_phrases), axis=0)
# 	idx_phrases 	= {k: idx for idx, k in enumerate(phrases)}
# 	doc_len 		= tf_terms.shape[0]

# 	# Wrap data
# 	data_dict = {
# 		'doc_len': doc_len,
# 		'phrases': phrases,
# 		'dfg_phrases': dfg_phrases,
# 		'idx_phrases': idx_phrases,
# 		'idx_terms': idx_terms,
# 		'tf_terms': tf_terms
# 	}

# 	# Run task
# 	res, pos_res = calc_ngram_tfidf(data_dict, 0, len(idx_phrases), method)
# 	sel_phrases  = np.array(phrases)[pos_res]

# 	return res, sel_phrases