
import numpy as np
import math
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

def get_mi(X, columns):
	mis = [0]*len(columns)
	n = len(X)
	j = 0
	for i in columns:
		_sum = 0
		for x in X:
			_sum += x[i]
		mis[j] = (_sum/n)
		j += 1
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
		#return np.cov(feature_2, feature_3)
		return np.cov(feature_1, feature_2)
	elif len(columns) == 3:
		v = np.vstack([feature_0, feature_1, feature_2])
		return np.cov(v)

def plot_multi_gaussian(X, Y, Z):
	X = np.asarray(X)
	Y = np.asarray(Y)
	Z = np.asarray(Z)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(X,Y,Z, linewidth=0)
	ax.contourf(X,Y,Z, zdir='z', offset=-0.15, cmap=cm.viridis)
	ax.set_xlabel("Carga")
	ax.set_ylabel("VO2 max")
	time.sleep(2)
	plt.show()
	plt.savefig('test.png')


def split_ages(X):
	id_1 = []
	id_2 = []
	id_3 = []

	for x in X:
		if x[0] > 40 and x[0] < 50: id_1.append(x)
		elif x[0] >= 50 and x[0] < 60: id_2.append(x)
		elif x[0] > 60 and x[0] < 70: id_3.append(x)

	return [id_1, id_2, id_3]

def format_parameters(x):
	mus = [[np.mean(x[0])], [np.mean(x[1])], [np.mean(x[2])]] 

	covariance_matrix = np.cov(x)
	return np.asmatrix(mus), np.asmatrix(covariance_matrix)

def predict_multi_gaussian(X, ids):
	mus_cov = []
	Y_multi_gauss_pred = []
	i = 0

	for d in ids:
		data = []
		for x in d:
			data.append([float(x[0]), float(x[1]), float(x[2])])

		data = np.asmatrix(normalize(data))  
		data = data.H 

	
		mus, covariance_matrix = format_parameters(data)

		print "Modelo ", i
		print "Mis: ", mus
		print "Sigma: ", covariance_matrix
		print "\n"
		i += 1

		inverse_covariance = covariance_matrix.I
		det_covariance = np.sqrt(np.linalg.det(covariance_matrix))
		d = 3


		result = []
		for x in data.H:
			value = (x - mus.H)*covariance_matrix.I*(x - mus.H).H 
			value_exp = np.exp((-1/2)*value)
			dem = (1/(np.power((2*math.pi), d/2)))*(1/det_covariance)
			result.append(value_exp/dem)
		Y_multi_gauss_pred.append(result)

	return Y_multi_gauss_pred

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

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, t, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(t)
    if label:
        ax.scatter(t[:, 0], t[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(t[:, 0], t[:, 1], s=40, zorder=2)
    ax.axis('equal')

    plt.show()


def mixture_gaussians(X, test, c):
	#Remove age column
	X = np.array(X)
	X = np.delete(X, 0, 1)
	test = [81.5, 181, 32.6]
	test = np.array(test)
	#test = np.delete(test,0,1)

	gmm = GaussianMixture(n_components = c)
	gmm = gmm.fit(X)

	print "Mixture of Gaussians parameters"
	print "Mis: ", gmm.means_
	print "Sigma: ", gmm.covariances_

	#return [max(x) for x in gmm.predict_proba(test.reshape(1,-1))]
	return gmm.predict_proba(test.reshape(1,-1))