
import pandas as pd 
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

class EF:
	''' This class holds the implementation of Expert Finder algorithm.

	Parameters
	----------
	ed_graph: nx.DiGraph
		An directed,  weightedbipartite graph (from publications to experts)

	et_matrix: np-array (N x M)
		A personalisation matrix of expert-topic associations (N: experts | M: topics)

	dt_matrix: np-array (N x M)
		A personlisation matrix of publication-topic associations (N: publications | M: topics)		

	lamb_e: float
		A lambda value of expert

	lamb_d: float
		A lambda value of publication

	max_iter: int
		A number of iterations when transforming authorities and hubs

	ed_count: np-array
		A vector containing number of publications per expert

	de_count: np-array
		A vector containing number of experts per publication
	'''
	def __init__(self, ed_graph, et_matrix, dt_matrix, lamb_e, lamb_d, max_iter, ed_count, de_count):
		# Validate params
		if not isinstance(ed_graph, type(nx.DiGraph())):
			raise TypeError("The directed graph must be of type nx.DiGraph, but is {}".format(type(ed_graph)))

		for matrix in [et_matrix, dt_matrix]:
			if not isinstance(matrix, pd.DataFrame):
				raise TypeError("Matrix argument(s) must be of type {}, but is {}".format(pd.DataFrame, type(matrix)))

		for lamb in [lamb_e, lamb_d]:
			if lamb < 0 or lamb > 1:
				raise ValueError("Lambda argument(s) must be between 0 and 1, but is {}".format(lamb))

		if not isinstance(max_iter, int):
			raise TypeError("Argument 'max_iter' must be of type int, but is {}".format(type(max_iter)))
		else:
			if (max_iter <= 0):
				raise ValueError("Argument 'max_iter' must be greater than 0, but is {}".format(max_iter))

		# Initialise fields
		self.ed_graph 	= ed_graph
		self.et_matrix 	= et_matrix
		self.dt_matrix 	= dt_matrix
		self.lamb_e 	= lamb_e
		self.lamb_d 	= lamb_d
		self.max_iter 	= max_iter
		self.ed_count 	= ed_count
		self.de_count 	= de_count

		self.ep_matrtix = pd.DataFrame(nx.to_numpy_matrix(ed_graph), index=ed_graph.nodes, columns=ed_graph.nodes)

	def transform(self, topic, init_h=None, norm='l2'):
		''' This function calculates authories and hubs given a sing topic

		Parameters
		----------
		topic: string
			A topic-search

		init_h: np-array (default: None)
			An initial hubs vector

		norm: string ('l1' or 'l2' - default: 'l2')
			A normalisation method

		Attributes
		----------
		pet_vec: np-array
			A personlisation vector of expert given a particular topic

		ppt_vec: np-array
			A personlisation vector of publication given a particular topic

		Return
		------
		H, A: np-array
			Vectors of hubs and authories respectively
		'''
		if topic is None:
			raise ValueError("Topic should not be None")
		if norm not in ['l1', 'l2']:
			raise ValueError("Normalisation method should be either l1 or l2, but received {}".format(norm))

		# Intialise
		A = None
		H = None

		# Initialise vectors and matrices
		if init_h is None:
			vec_shape = (len(self.ed_graph.nodes()), 1)
			init_h = np.ones(vec_shape)
		trans_matrix = self.ep_matrtix.values.T.copy()

		# Process personalised vector of expert and publication nodes
		pet_vec = self.et_matrix[topic].values.reshape(-1, 1).copy().astype(np.float32)
		ppt_vec = self.dt_matrix[topic].values.reshape(-1, 1).copy().astype(np.float32)

		# Normalisation
		pet_vec = normalize(pet_vec, norm=norm, axis=0)
		ppt_vec = normalize(ppt_vec, norm=norm, axis=0)

		# HITS algorithm
		for _ in range(self.max_iter):
			Ag_struct = np.divide(np.matmul(trans_matrix, init_h), self.ed_count.values)
			A = np.sum([
				pet_vec.dot(1 - self.lamb_e),
				Ag_struct.dot(self.lamb_e)
			], axis=0)

			Hg_struct = np.divide(np.matmul(self.ep_matrtix.values, A), self.de_count.values)
			H = np.sum([
				ppt_vec.dot(1 - self.lamb_d),
				Hg_struct.dot(self.lamb_d)
			], axis=0)

			A = normalize(A, norm=norm, axis=0)
			H = normalize(H, norm=norm, axis=0)

			pet_vec = A
			ppt_vec = H
			init_h = H

		return H, A