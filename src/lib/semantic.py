
import torch

class PhraseVectorizer:
	''' This class contains the implementation of the phrase embedding process
	with the given model and tokenizer

	Parameters
	----------
	model: BertModel
		A pretrained embedding model
	tokenizer: BerTokenizer
		A pretrained tokenizer model
	'''
	def __init__(self, model, tokenizer):
		self.model 		= model
		self.tokenizer  = tokenizer

	def transform(self, phrase):
		''' This function transform a given phrase into a numpy vector
		
		Parameters
		----------
		phrase: str
			A single phrase

		Return
		------
		np_vec: np.ndarray
			A numpy vector of a given phrase
		'''
		# Encoding-Decoding tensor
		encoded_phrase = self.tokenizer.encode(phrase)
		in_tensor = torch.tensor(encoded_phrase).unsqueeze(0)
		out_tensor = self.model(in_tensor)
		last_hidden_states = out_tensor[0]

		# Convert to numpy vector
		np_vec = last_hidden_states.mean(1).detach().numpy()
		return np_vec
