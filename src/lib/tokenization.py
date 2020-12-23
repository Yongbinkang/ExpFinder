import re
from nltk.stem import WordNetLemmatizer
import nltk

class Tokenizer:
	''' This class contains the implementation for noun phrase tokenisation

	Parameters
	----------
	stopwords: set(str)
		A set of stopwords
	
	Attributes
	----------
	lemmatizer: nltk.stem.WordNetLemmatizer
		A lemmatizer model from NLTK library

	pos_family: dict
		A dictionary containing part of speech (pos) and its family

	np_parser: nltk.RegexpParser
		A parser model from NLTK library

	digit_pattern: re.Complier
		A digit pattern compiled from re library

	word_pattern: Regex str
		A pattern for extracting each word

	Notes
	-----
	1. The word pattern includes following things:
		1.1. Set flag to allow verbose regexps
		1.2. Abbreviations, e.g. U.S.A.
		1.3. Words with optional internal hyphens
		1.4. Currency and percentages, e.g. $12.40, 82%
		1.5. Ellipsis
		1.6. These are separate tokens; includes ], [
	'''
	def __init__(self, stopwords):
		self.stopwords = stopwords
		self.lemmatizer = WordNetLemmatizer()
		self.pos_family = {
			'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
			'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
			'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
			'adj': ['JJ', 'JJR', 'JJS'],
			'adv': ['RB', 'RBR', 'RBS', 'WRB']
		}
		self.np_parser = self._build_np_parser()    	 
		self.digit_pattern = re.compile(r"[\d/-]+$")
		self.word_pattern = r'''(?x)
			(?:[A-Z]\.)+
			| \w+(?:-\w+)*
			| \$?\d+(?:\.\d+)?%?
			| \.\.\.
			| [][.,;"?():-_`]
		'''

	def transform(self, doc, max_phrase_length=1):
		''' This function extracts tokens, nouns and noun phrases from a given document.

		Parameters
		----------
		doc: str
			A document in a corpus
		max_phrase_length: int (default = 1)
			An inclusive maximum lengh of a particular phrase
		
		Return
		------
		F: dict
			A dictionary of statistical information about the document
		T: dict
			A dictionary of tokens, nouns and noun phrases
		'''
		if not isinstance(doc, str):
			raise TypeError("doc should be type of str")
		if not isinstance(max_phrase_length, int):
			raise TypeError("max_phrase_length should be type of int")

		# Intialise
		noun_count = 0
		pron_count = 0
		verb_count = 0
		adj_count = 0
		adv_count = 0

		stopword_count = 0
		char_count = 0
		char_count = 0
		unique_words = set()
		word_count = 0

		tokens = []
		nouns = []
		bigrams = []
		noun_phrases = []
		cfinder_phrases = []

		# Parsing process
		sentences = nltk.sent_tokenize(doc)
		for s in sentences:
			s = s.lower()
			t = str.maketrans(dict.fromkeys("'`", ""))
			s = s.translate(t)
			words = nltk.regexp_tokenize(s, pattern=self.word_pattern)
			pairs = nltk.pos_tag(words)
			if len(words)==0: continue

			# Extrating tokens and nouns with statistical information
			for pair in pairs:
				tag = list(pair)[1]
				w = list(pair)[0]
				if self.digit_pattern.match(w): continue

				if tag in self.pos_family['noun']:
					w = self.lemmatizer.lemmatize(w, 'n')
					if len(w) <= 2: continue
					if w in self.stopwords:
						stopword_count += 1
						continue
					noun_count += 1
					nouns.append(w)
					tokens.append(w)
					char_count += len(w)
					word_count += 1
					unique_words.add(w)
				elif tag in self.pos_family['pron']:
					pron_count += 1
					if len(w) <= 2: continue
					if w in self.stopwords:
						stopword_count += 1
						continue
					char_count += len(w)
					word_count += 1
					unique_words.add(w)
					tokens.append(w)
				elif tag in self.pos_family['verb']:
					verb_count += 1
					if len(w) <= 2: continue
					if w in self.stopwords:
						stopword_count += 1
						continue
					char_count += len(w)
					word_count += 1
					unique_words.add(w)
					tokens.append(w)
				elif tag in self.pos_family['adj']:
					adj_count += 1
					w = self.lemmatizer.lemmatize(w, 'a')
					if len(w) <= 2: continue
					if w in self.stopwords:
						stopword_count += 1
						continue
					tokens.append(w)
					char_count += len(w)
					word_count += 1
					unique_words.add(w)
				elif tag in self.pos_family['adv']:
					adv_count += 1
					if len(w) <= 2: continue
					if w in self.stopwords:
						stopword_count += 1
						continue
					char_count += len(w)
					word_count += 1
					unique_words.add(w)
					tokens.append(w)
		
			# Extracting noun phrases based on the designed grammar
			tree = self.np_parser.parse(pairs)
			for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
				phrases = []
				for w, pos in subtree.leaves():
					if self.digit_pattern.match(w): continue

					if pos.startswith('N'):
						w = self.lemmatizer.lemmatize(w, 'n')
						if len(w) < 2: continue
						if w in self.stopwords: continue
						if len(phrases) < max_phrase_length:
							phrases.append(w)
						else:
							noun_phrases.append('_'.join(phrases))
							phrases = []
							phrases.append(w)
					elif pos.startswith('J'):
						w = self.lemmatizer.lemmatize(w, 'a')
						if len(w) < 2: continue
						if w in self.stopwords: continue
						if len(phrases) < max_phrase_length:
							phrases.append(w)
						else:
							if (len(phrases) == 1):
								if (len(phrases[0]) <= 2): continue
							noun_phrases.append('_'.join(phrases))
							phrases = []
							phrases.append(w)
					else:
						if len(w) < 2: continue
						if w in self.stopwords: continue
						if len(phrases) < max_phrase_length:
							phrases.append(w)
						else:
							if (len(phrases) == 1):
								if (len(phrases[0]) <= 2): continue
							noun_phrases.append('_'.join(phrases))
							phrases = []

				if len(phrases) > 0 and len(phrases) <= max_phrase_length:
					if (len(phrases)==1):
						if (len(phrases[0]) <=2): continue
					noun_phrases.append('_'.join(phrases))

		# Gneral statistical calculation
		unique_word_count = len(unique_words)
		word_density = char_count / (word_count + 1)
		sent_count = len(sentences)
		title_word_account = len(doc.split(".")[0].split())

		# Construct the return
		F = {
			'noun_count': noun_count,
			'pron_count': pron_count,
			'verb_count': verb_count,
			'adj_count': adj_count,
			'adv_count': adv_count,
			'word_count': word_count,
			'unique_word_count': unique_word_count,
			'word_density': word_density,
			'sent_count': sent_count,
			'title_word_account': title_word_account,
			'stopword_count': stopword_count
		}

		T = {
			'tokens': ' '.join(tokens),
			'nouns': ' '.join(nouns),
			'np': ' '.join(noun_phrases)
		}
		return F, T


	def _build_np_parser(self):
		noun_grammar = r'''
			NP: {<NN.*|JJ.*|VBN.*|VBG.*>*<NN.*>}
				{<NNP>+}
			'''
		return nltk.RegexpParser(noun_grammar)