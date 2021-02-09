
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from src.lib import vectorizer, weight
import networkx as nx

#################### MATRIX ########################

def generate_ed_matrix(ed_df):
	''' This function generates the expert-document matrix

	Parameters
	----------
	ed_df: pd.DataFrame
		An expert-document dataframe where columns are exp_id, doc_id and weight

	Return
	------
	ep_matrix: pd.DataFrame
		An expert-document matrix
	'''
	# Validation
	for col in ed_df.columns:
		if col not in ['doc_id', 'exp_id', 'weight']:
			raise ValueError("ed_df columns should contains only exp_id, doc_id and weight")

	# Initialise
	exp_ids = ed_df['exp_id'].unique()
	doc_ids = ed_df['doc_id'].unique()

	# Crosstab matrix
	ep_matrix = pd.crosstab(ed_df['exp_id'], ed_df['doc_id']).values

	return pd.DataFrame(ep_matrix, index=exp_ids, columns=doc_ids)

def generate_tf(corpus, encoding='utf-8', token_pattern=r"(?u)\S\S+"):
	''' This function generates the unnormalized TF weight of a given corpus

	Parameters
	----------
	corpus: list(str)
		A list of documents
	encoding: str
		If bytes or files are given to analyze, this encoding is used to decode.
	token_pattern: Regex str
		Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'.

	Return
	------
	A dictionary of TF model and weight
		model: Sklearn CountVectorizer
			A TF model for transforming the TF weight for a corpus
		trans: Scipy csr_matrix
			A TF weight of all single word in a corpus
	'''
	# Count Vectorization
	model = CountVectorizer(analyzer='word', encoding=encoding, token_pattern=token_pattern)
	weight = model.fit_transform(corpus)
	return { 'model': model, 'trans': weight }

def generate_dp_matrix(tf_terms, tf_phrases, doc_ids, method="indirect"):
	''' This function generates the nTFIDF for phrases based on the given TF 
	information of terms and phrases.

	Parameters
	----------
	tf_terms: dict
		A dictionary containing TF information of terms (including model and tranformation)
	tf_phrases: dict
		A dictionary containing TF information of phrases (including model and tranformation)
	doc_ids: array_like	
		A list of documents' id
	method: str
		A calculation method for nIDF ("indirect" - default | "direct")
	
	Return
	------
	res: dict
		A dictionary containing index, columns and nTFIDF matrix 
	'''
	# Preparing data
	model_dict = {
		'tft_model': tf_terms['model'],
		'tft_trans': tf_terms['trans'],
		'tfp_model': tf_phrases['model'],
		'tfp_trans': tf_phrases['trans']
	}

	# Run nTFIDF tasks
	tfidf_ngrams, phrases = weight.calc_ngram_tfidf(model_dict, method)

	# Wrap result
	res = {
		'index': doc_ids,
		'columns': phrases,
		'matrix': csr_matrix(tfidf_ngrams)
	}
	return res

def generate_dtop_matrix(dp_matrix, topics, model_dict, top_n=1):
	''' This function generates the document-topic matrix based on top-n similar phrases 
	from the given document-phrase matrix with pretrained model.
	
	Parameters
	----------
	dp_matrix: pd.DataFrame
		A document-phrase matrix
	topics: array_like
		A list of topics
	model_dict: dict
		A dictionary containing pretrained embedding model, tokenizer and trained vectors (optional)
	top_n: int
		A number of top similar phrases for each given topic

	Return
	------
	res: dict
		A dictionary containing index, columns and matrix of the document-topic matrix
	topic_phrase: dict
		A dictionary containing information between topic and phrase for debugging
	'''
	# Parse model
	model = model_dict['model']
	tokenizer = model_dict['tokenizer'] 
	trained_vecs = model_dict['trained_vectors']

	# Initialise
	phrases = dp_matrix.columns
	topic_scores = []
	topic_phrase = dict()
	pVectorizer = vectorizer.PhraseVectorizer(model, tokenizer)
	

	# Embed phrases into vectors (if there are trained ones, load it instead)
	if trained_vecs is None:
		phrase_vecs = []
		for phrase in phrases:
			temp_p = ' '.join(phrase.split('_'))
			if temp_p == '':
				continue
			phrase_vecs.append(pVectorizer.transform(temp_p)[0])
	else:
		phrase_vecs = trained_vecs

	# Calculate weight for each topic with top-n similar phrases
	for topic in topics:		
		sim_phrases = []
		sim_vals = []
		topic_vec = pVectorizer.transform(topic)

		# Find top-n phrase
		scores = cosine_similarity(topic_vec, phrase_vecs)[0]
		scores = {idx: val for idx, val in enumerate(scores)}
		scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

		for score in scores[:top_n:]:
			idx, val = score
			sim_phrases.append(phrases[idx])
			sim_vals.append(val)

		# Calculate weight for a topic
		filtered_matrix = dp_matrix[sim_phrases].values
		num = np.sum(np.multiply(filtered_matrix, sim_vals), axis=1)
		demon = np.sum(sim_vals)
		avg_score = np.divide(num, demon)
		topic_scores.append(avg_score)

		# Construct topic_phrase dictionary
		topic_phrase[topic] = [(sim_phrases[i], sim_vals[i]) for i in range(len(sim_phrases))]

	# Normalisation
	topic_scores = normalize(np.array(topic_scores), norm='l1', axis=0)

	# Wrap result
	res = {
		'index': dp_matrix.index,
		'columns': topics,
		'matrix': csr_matrix(topic_scores.T)
	}
	return res, topic_phrase

def generate_pr_matrix(ed_matrix, dtopic_matrix, topic_vec, ed_graph, alpha=0.0):
	''' This function generates the personlisation matrices of expert and document
	based on the given expert-document graph

	Paramaeters
	-----------
	ed_matrix: pd.DataFrame
		An expert-document matrix
	dtopic_matrix: pd.DataFrame
		A document-topic matrix
	topic_vec: np.ndarray
		A 1-dim array containing weights for each topic		
	ed_graph: nx.DiGraph
		An expert-document graph
	alpha: int or float (alpha in [0, 1])
		An alpha value for smoothing factor		

	Return
	------
	res_exp: dict
		A dictionary containing index, columns and matrix of expert-topic
	res_doc: dict 
		A dictionary containing index, columns and matrix of document-topic
	'''
	# Initialise
	exp_pr = []
	doc_pr = []
	exp_ids = ed_matrix.index
	doc_ids = dtopic_matrix.index
	zero_vec = np.zeros(len(dtopic_matrix.columns), dtype=np.float32)

	# Calculate personalisation
	exp_pr_df, doc_pr_df = weight.calc_pr_weight(ed_matrix, dtopic_matrix, topic_vec, alpha)

	# Mapping nodes
	for node in ed_graph.nodes():
		if node in exp_ids:
			temp = np.array(exp_pr_df.loc[node].values, dtype=np.float32)
			exp_pr.append(temp)
			doc_pr.append(zero_vec)
		elif node in doc_ids:
			temp = np.array(doc_pr_df.loc[node].values, dtype=np.float32)
			exp_pr.append(zero_vec)
			doc_pr.append(temp)

	# Wrap results
	res_exp = {
		'index': ed_graph.nodes(),
		'columns': exp_pr_df.columns,
		'matrix': csr_matrix(exp_pr)
	}
	res_doc = {
		'index': ed_graph.nodes(),
		'columns': exp_pr_df.columns,
		'matrix': csr_matrix(doc_pr)
	}
	return res_exp, res_doc

#################### GRAPH ########################
def generate_ecg(ed_df):
	''' This function is to build an ECG graph given associations of expert
	and documents in the pandas DataFrame

	Parameters
	----------
	ed_df: pd.DataFrame
		An Expert-Document dataframe which contains two columns `doc_id` and `exp_id`

	Return
	------
	G: nx.DiGraph
		A directed, weighted bipartite graph whose directions are from documents to experts
	'''
	# Validate parameters
	if 'doc_id' not in ed_df.columns:
		raise ValueError('en_df must contain the `doc_id` column')
	if 'exp_id' not in ed_df.columns:
		raise ValueError('en_df must contain the `exp_id` column')
	
	# Build graph
	edges = ed_df[['doc_id', 'exp_id']].values
	G = nx.DiGraph()
	G.add_edges_from(edges, weight=1, length=0.05)

	return G


#################### VECTOR ########################

def generate_topic_vector(dtopic_matrix):
	''' This functio generates a topic vector based on the given 
	document-topic matrix

	Parameters
	----------
	dtopic_matrix: dict
		A dictionary containing information of document-topic matrix (columns, index and matrix)

	Return
	------
	topic_vec: pd.DataFrame
		A topic vector
	'''
	# Initialise
	topics = dtopic_matrix['columns']
	D = dtopic_matrix['matrix']

	# Count number of docs
	num_docs = D.shape[0]

	# Averaging with weights of documnet
	topic_vec = np.sum(D, axis=0) / num_docs

	return pd.DataFrame(topic_vec.T, index=topics, columns=['weights'])

def generate_ed_vector(ed_matrix, ed_graph):
	''' This function generates the counted vectors of experts and documents.
	For example, the counted vector of experts contains number of document associated
	to each particular expert

	Parameters
	----------
	ed_matrix: pd.DataFrame
		An expert-document matrix
	ed_graph: nx.DiGraph
		An expert-document graph

	Return
	------
	exp_vec: pd.DataFrame
		A counted-document experts vector
	doc_vec: pd.DataFrame
		A counted-experts document vector 
	'''
	# Intialise
	exp_vec = []
	doc_vec = []
	exp_ids = ed_matrix.index
	doc_ids = ed_matrix.columns

	# Mapping nodes
	for node in ed_graph.nodes():
		if node in exp_ids:
			temp = np.sum(ed_matrix.loc[node].values)
			if temp == 0:
				exp_vec.append(1)
			else:
				exp_vec.append(temp)
			doc_vec.append(1)

		elif node in doc_ids:
			temp = np.sum(ed_matrix[node].values)
			if temp == 0:
				doc_vec.append(1)
			else:
				doc_vec.append(temp)
			exp_vec.append(1)
	
	exp_vec = pd.DataFrame(
		np.array(exp_vec).reshape(-1, 1),
		index=ed_graph.nodes, columns=['count']
	)
	doc_vec = pd.DataFrame(
		np.array(doc_vec).reshape(-1, 1),
		index=ed_graph.nodes, columns=['count']
	)
	return exp_vec, doc_vec