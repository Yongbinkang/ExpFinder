''' 
This file contains the script for showing the example of the
phrase extraction using different linguistic patterns based on 
POS tags.
'''
from ast import literal_eval
from src.lib import np_extractor

def main():
	test_sample = ["An embedded system is a computer system that has a dedicated function within a larger mechanical or electrical system. Embedded systems are often based on microcontrollers, but ordinary microprocessors are also common, especially in more complex systems."]

	# Load stopwords
	with open('./data/stopword.txt') as f:
		stopwords = literal_eval(f.read())

	# Extract phrases with the default linguistic pattern (JJ)*|(VBN)*|(VBG)*(N)+
	default_res = np_extractor.extract_np(test_sample, stopwords, max_phrase_len=3)

	# Extract phrases with the new linguistic pattern  (JJ)*(N)+
	pattern = r'''
	NP: {<NN.*|JJ.*>*<NN.*>}
		{<NNP>+}
	'''
	new_res = np_extractor.extract_np(test_sample, stopwords, max_phrase_len=3, pattern=pattern)

	# Present the result
	print("Input:\n{}".format(test_sample))
	print("Output (default pattern): {}".format(default_res['np'][0].split()))
	print("Output (new pattern): {}".format(new_res['np'][0].split()))

if __name__ == "__main__":
	main()