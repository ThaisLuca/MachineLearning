
import numpy as np 
import random

def get_cetroids(data, K):
	return random.sample(data, K)

def k_means(data, K):

	#Initialize clusters
	mu = get_cetroids(data, K)