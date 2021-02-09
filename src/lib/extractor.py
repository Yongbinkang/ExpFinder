
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