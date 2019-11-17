import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

def get_correct_classification(X):
	classes = []
	for x in X:
		if int(x[0]) >= 18 and int(x[0]) < 40: classes.append(1)
		elif int(x[0]) >= 40 and int(x[0]) < 60: classes.append(2)
		elif int(x[0]) >= 60: classes.append(3)
	return classes

def get_probabilities(ids):
	size = len(ids[0])+len(ids[1])+len(ids[2])
	pi_id1 = len(ids[0])/size
	pi_id2 = len(ids[1])/size
	pi_id3 = len(ids[2])/size

	return [pi_id1, pi_id2, pi_id3]

def ages_nb(ids):
	_id1 = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in ids[0]]
	_id2 = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in ids[1]]
	_id3 = [[float(i[0]), float(i[1]), float(i[2]), float(i[3])] for i in ids[2]]

	return [np.asmatrix(_id1), np.asmatrix(_id2), np.asmatrix(_id3)]

def multivariate_gaussian(X, mu, covariance, dimension):
	res = []
	mu = np.asmatrix(mu)
	covariance = np.asmatrix(covariance)
	inverse_covariance = covariance.I
	det_covariance = np.sqrt(np.linalg.det(covariance))
	for x in X:


		#MODIFICAR ANTES DE ENVIAR

		value = (x - mu.H).H*covariance.I*(x - mu.H)
		value_exp = np.exp((-1/2)*value)
		dem = (1/(np.power((2*math.pi), dimension/2)))*(1/det_covariance)
		res.append(value_exp/dem)
	return res

def get_mis(_id):
	return [np.mean(_id.H[1]), np.mean(_id.H[2]), np.mean(_id.H[3])]

def get_covariance(_id):
	_id_t = np.asmatrix(_id.H[1:])
	cov = np.cov(_id_t)
	return np.asmatrix(cov)


def naive_bayes(train, ids, test):
	d = 3

	# Computes all values for pi for each class
	pis = get_probabilities(ids)

	# Find the mis for each feature by class
	mis_1 = get_mis(ids[0])
	mis_2 = get_mis(ids[1])
	mis_3 = get_mis(ids[2])

	# Find the covariances for each feature by class
	cov_1 = get_covariance(ids[0])
	cov_2 = get_covariance(ids[1])
	cov_3 = get_covariance(ids[2])

	test = np.asmatrix(normalize(test))

	#Predict for each class
	#First class
	y_mult_1 = multivariate_gaussian(test, mis_1, cov_1, d)
	y_1 = []
	for i in y_mult_1:
		y_1.append([pis[0]*i.item(0)])

	#Second class
	y_mult_2 = multivariate_gaussian(test, mis_2, cov_2, d)
	y_2 = []
	for i in y_mult_2:
		y_2.append(pis[1]*i.item(0))

	#Third class
	y_mult_3 = multivariate_gaussian(test, mis_3, cov_3, d)
	y_3 = []
	for i in y_mult_3:
		y_3.append(pis[2]*i.item(0))


	y_pred = []
	for i in range(0, len(y_1)):
		max_ = np.max([y_1[i], y_2[i], y_3[i]])
		if y_1[i] == max_: y_pred.append(1)
		elif y_2[i] == max_: y_pred.append(2)
		elif y_3[i] == max_: y_pred.append(3)

	return y_pred




