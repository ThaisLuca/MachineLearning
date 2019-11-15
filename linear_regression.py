
import numpy as np
import matplotlib.pyplot as plt 

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
			if(feature_3 != 0): row.append(np.power(x[feature_3], d))
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
	plt.scatter(X, Y)
	plt.plot(X, Y_pred, color='red')
	plt.show()

def linear_regression(X, Y, degree, index):
	# Calculates fi(x) for one feature linear regression
	fi = get_fi_uni(X, degree, index)

	# Returns weights
	w = get_weights(fi, Y)

	return predict(X, w, degree, index, 0, 0)

def multiple_linear_regression(X, Y, degree, feature_1, feature_2, feature_3):
	# Calculates fi(x) for multiples features
	fi = get_fi_multi(X, degree, feature_1, feature_2, feature_3)

	# Returns weights
	w = get_weights(fi, Y)

	return predict(X, w, degree, feature_1, feature_2, feature_3, True)

