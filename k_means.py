
from __future__ import division
import numpy as np 
import random
from sklearn.cluster import KMeans


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

def get_age(example):
	# [18-30)
	if 18 <= example[0] < 30:
		return 0
	# [30-50)
	if 30 <= example[0] < 50:
		return 1
	# [50-60)
	if 50 <= example[0] < 60:
		return 2
	# [60-70)
	if 60 <= example[0] < 70:
		return 3
	# [70-80)
	if 70 <= example[0] < 80:
		return 4
	# [80-100)
	if 80 <= example[0] < 100:
		return 5

def get_fraction(data, r, K):
	
	for k in range(0, K):
		ages = [0]*6
		for i in range(0, len(data)):
			if r[i][k] == 1:
				f = get_age(data[i])
				ages[f] += 1
				
		for f in range(0, len(ages)):
			if ages[f] != 0:
				ages[f] /= len(data)
		print "Cluster ", k
		print ages
		print "\n"


def k_means(data, K):
	n_iterations = 50
	#data = data[:5]

	#Initialize clusters
	# r is used for 1-of-K code scheme
	# a point belongs to a cluster if r[i][cluster_number] = 1
	# mu is the array for storing centroids
	r, mu = get_cetroids(data, K)

	for n in range(0, n_iterations):
		last_mu = mu
		r = optimize_r(data, mu, r, K)
		mu = optimize_mu(data, mu, r, K)

		#If centroids don't change, break
		if mu == last_mu: break
		erro = objective_function(r, data, mu, K)

	print "Centroids: ", mu
	print "\n"
	print "Localization for each example: ", r

	print "\n"
	print "Python K-Means"
	kmeans = KMeans(n_clusters=K).fit(data)
	centroids = kmeans.cluster_centers_
	print centroids

	print "\n"
	print "Cluster fraction by age:"
	get_fraction(data, r, K)