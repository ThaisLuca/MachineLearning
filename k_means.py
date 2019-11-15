
import numpy as np 
import random

def find_centroid_index(centroids, data):
	i = 0
	for c in centroids:
		if c == data:
			return i
		else:
			i += 1
	# 404 centroid not found
	return None

def get_cetroids(data, K):
	centroids = random.sample(data, K)

	indexes = np.zeros((len(data), K))
	for i in range(0, len(data)):
		j = find_centroid_index(centroids, data[i])
		if j != None:
			indexes[i][j] = 1

	return indexes, centroids


def optimize_r(data, mu, r, K):
	diffs = []
	for i in range(0, len(data)):
		min_j = None
		minimum = None
		
		for k in range(0, K):
			diff = [data[i][t]-mu[k][t] for t in range(0, len(data[i]))]
			diff = np.linalg.norm(diff)**2
			if min_j == None or minimum > diff:
				min_j = k
				minimum = diff
		r[i][min_j] = 1
	return r

def optimize_mu(data, mu, r, K):
	for k in range(0, K):
		r_x = [0]*len(data[0])
		r_x = np.array(r_x)
		r_k = 0
		for i in range(0, len(data)):
			if r[i][k]: 
				d = np.array(data[i])
				r_x = r_x + d
			r_k += r[i][k]
		mu[k] = r_x / r_k
	return mu

def objective_function(r, X, mu, K):
	_sum = 0
	for i in range(0, len(X)):
		for k in range(0,K):
			diff = [X[i][t]-mu[k][t] for t in range(0, len(X[i]))] 
			_sum += r[i][k] * np.linalg.norm(diff)
	return _sum

def k_means(data, K):
	n_iterations = 20

	#Initialize clusters
	# indexes is used for 1-of-K code scheme
	data = data[:5]
	r, mu = get_cetroids(data, K)

	for n in range(0, n_iterations):
		r = optimize_r(data, mu, r, K)
		mu = optimize_mu(data, mu, r, K)
		erro = objective_function(r, data, mu, K)
		print "Erro ", erro

	# print "Centroids: ", mu
	# print "Localization for each example: ", r
	
	


	