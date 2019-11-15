
import numpy as np
import math
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_mi(X, columns):
	mis = [0]*len(columns)
	n = len(X)
	for i in range(0, len(columns)):
		_sum = 0
		for x in X:
			_sum += x[i]
		mis[i] = (_sum/n)
	return mis

def get_variance(X, mi):
	n = len(X)
	diff = X - mi
	return (diff*diff.H)/n

def get_covariance_matrix(X, columns):
	X = np.array(X)
	feature_0 = np.asmatrix(X[:,0])
	feature_1 = np.asmatrix(X[:,1])
	feature_2 = np.asmatrix(X[:,2])
	feature_3 = np.asmatrix(X[:,3])
	if len(columns) == 2: 
		return np.cov(feature_2, feature_3)
	elif len(columns) == 3:
		v = np.vstack([feature_1, feature_2, feature_3])
		return np.cov(v)
	else:
		v = np.vstack([feature_0, feature_1, feature_2, feature_3])
		return np.cov(v)
	return cov

def plot_multi_gaussian(X, Y, Z):
	X = np.asarray(X)
	Y = np.asarray(Y)
	Z = np.asarray(Z)

	print len(X), len(Y), Z.shape

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(X,Y,Z, linewidth=0)
	ax.contourf(X,Y,Z, zdir='z', offset=-0.15, cmap=cm.viridis)
	ax.set_xlabel("Carga")
	ax.set_ylabel("VO2 max")
	time.sleep(2)
	plt.show()
	plt.savefig('test.png')
	

def multivariate_gaussian(X, mi, sigma):
	size = len(X)
	
	mi = np.asmatrix(mi)
	sigma = np.asmatrix(sigma)

	mi_t = mi.H
	det = np.linalg.det(sigma)
	inv = sigma.I

	sqrt_ = math.sqrt(((math.pi*2)**2)*det)
	pp = 1.0/sqrt_
	diff = (X - mi)
	fac = diff*inv*diff.H
	sp = np.exp(-(1/2)*(fac))
	return pp*sp