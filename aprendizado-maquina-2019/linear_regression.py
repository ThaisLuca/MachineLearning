
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import linear_model

# fi has the form fi(x) = [1, x, x^2,..., x^d]
def get_fi_uni(data, degree, index):
	fi =  []
	row = []
	for x in data:
		for d in range(0, degree):
			row.append(np.power(x[index], d))
		fi.append(row)
		row = []
	return fi

# fi for multiple features has form fi(x) = [1, x_1, x_2, ..., x_1^d, x_2^d]
def get_fi_multi(data, degree, feature_1, feature_2, feature_3):
	fi = []
	row = [1]
	for x in data:
		for d in range(1, degree+1):
			row.append(np.power(x[feature_1], d))
			row.append(np.power(x[feature_2], d))
			#row.append(np.power(x[feature_3], d))
		fi.append(row)
		row = [1]
	return fi

def get_weights(X, Y):
	X = np.asmatrix(X)
	Y = np.asmatrix(Y)

	inverse = (X.H*X).I 
	w = inverse*X.H*Y

	return w

def predict(X, w, degree, feature_1, feature_2, feature_3, multiple=False):
	w = np.asmatrix(w)
	if multiple:
		X = get_fi_multi(X, degree, feature_1, feature_2, feature_3)
	else:
		X = get_fi_uni(X, degree, feature_1)
	return X*w

def negative_log_likelihood(y_pred, Y):
	error = Y - y_pred
	return (error.H*error)/2


def plot_linear_regression(X, Y, Y_pred):
	plt.xlabel("Carga")
	plt.ylabel("VO2 max")
	plt.scatter(X, Y)
	plt.plot(X, Y_pred, color='red')
	plt.show()

def plot_multi_linear_regression(X, Y, Z):

	# x = np.linspace(max(X),max(X),10)
	# y = np.linspace(min(Y),max(Y),10)

	# X_,Y_ = np.meshgrid(X,Y)
	# Z=0.12861723162963065*X_ + 0.0014024845304814665*Y_ + 0.0964608113924048

	# plot the surface
	plt3d = plt.figure().gca(projection='3d')
	plt3d.plot_surface(X, Y, Z, linewidth=0)
	plt.xlabel("Carga")
	plt.ylabel("Peso")

	# Ensure that the next plot doesn't overwrite the first plot
	ax = plt.gca()

	ax.scatter(X, Y, color='blue')
	plt.show()

def linear_regression(X, Y, degree, index):
	# Calculates fi(x) for one feature linear regression
	fi = get_fi_uni(X, degree, index)

	# Returns weights
	w = get_weights(fi, Y)
	print "Weights: "
	print w

	return predict(X, w, degree, index, 0, 0)

def multiple_linear_regression(X, Y, degree, feature_1, feature_2, feature_3):
	# Calculates fi(x) for multiples features
	fi = get_fi_multi(X, degree, feature_1, feature_2, feature_3)

	# Returns weights
	w = get_weights(fi, Y)
	print "Weights"
	print w

	return predict(X, w, degree, feature_1, feature_2, feature_3, True)

