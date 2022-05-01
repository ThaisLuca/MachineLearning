import random
import numpy as np
from linear_regression import *
from gaussian import *
from sklearn.metrics import mean_squared_error
from k_means import *
import os.path
from naive_bayes import *


def load_data(filename):
	file = open(filename)
	data = file.readlines()
	return data


def split_train_test(data):

	if os.path.isfile('array.txt') :
		with open('array.txt', 'r') as f:
			indexes = f.readlines()
			for i in range(0, len(indexes)):
				indexes[i] = int(indexes[i].split()[0])
		f.close()
	else:
		train_sample = random.sample(range(len(data)), 1000)
		train_sample.sort()

		test_sample = []
		for i in range(0, len(data)):
			if i not in train_sample: test_sample.append(i)

		indexes = [0]*len(data)
		for i in train_sample:
			indexes[i] = 1

		with open('array.txt', 'w') as f:
			for item in indexes:
				f.write("%s\n" % item)
		f.close()

	train = []
	test = []
	for i in range(0, len(indexes)):
		if indexes[i] == 1: 
			train.append(data[i])
		else:
			test.append(data[i])

	return train, test


def prepare_data(data):
	matrix = []
	row = []
	for item in data:
		split = item.split()
		for i in range(0, len(split)):
			try:
				row.append(int(split[i]))
			except:
				row.append(float(split[i]))
		matrix.append(row)
		row = []
	return matrix

def get_column_matrix(data, index):
	d = []
	for x in data:
		d.append([x[index]])
	return d

def get_array(data, index):
	d = []
	for x in data:
		d.append(x[index])
	return d

def american_college_sports_equation(X):
	y = []
	for x in X:
		y.append((x[1]*11.4) + 260+(x[0]*3.5))
	return y 

def main():

	train, test = split_train_test(load_data('dados.txt'))
	train = prepare_data(train)
	test = prepare_data(test)
	#test = np.asmatrix(test)

	data = load_data('dados.txt')
	data = prepare_data(data)

	#Parameters for simple linear regression
	degree = 2
	x_column = 2
	answer_index = 3

	#Compute results for simple Linear Regression
	Y_linear_regression = get_column_matrix(train, answer_index)
	Y_pred_linear_regression = linear_regression(train, Y_linear_regression, degree, x_column)
	print "Negative log likelihood for simple linear regression: "
	print negative_log_likelihood(Y_pred_linear_regression, Y_linear_regression)
	print "\n"

	# plot regression line using train examples
	X_linear_regression = get_column_matrix(train, x_column)
	plot_linear_regression(X_linear_regression, Y_linear_regression, Y_pred_linear_regression)

	#Parameters for Multi Linear Regression
	degree = 10
	feature_1 = 0
	feature_2 = 1
	feature_3 = 2
	answer_index = 3

	X = np.array(train)

	# Compute results for Multiple Linear Regression
	Y_multilinear_regression = get_column_matrix(train, answer_index)
	Y_pred_multilinear_regression = multiple_linear_regression(train, Y_multilinear_regression, degree, feature_1, feature_2, feature_3)
	print "Negative log likelihood for multi linear regression: "
	print negative_log_likelihood(Y_pred_multilinear_regression, Y_multilinear_regression)
	print "\n"


	print "Predicted values using American College os Sports Equation:"
	a = american_college_sports_equation(train)
	print "\n"

	print mean_squared_error(Y_pred_multilinear_regression, a)

	plot_multi_linear_regression(get_array(train,2), get_array(train, 3), Y_pred_multilinear_regression)

	#Parameters for Multivariate Gaussian for 2 features
	columns = [2,3]

	#Find parameters
	mi = get_mi(train, columns)
	sigma = get_covariance_matrix(train, columns)
	print "Multivariate Gaussian Parameters using 2 features:"
	print "Mi: ", mi, "Sigma: ", sigma
	print "\n"
	X = np.array(train)
	Y_multivariate_gaussian = multivariate_gaussian(X[:,columns], mi, sigma)

	Y_multivariate_gaussian = multivariate_gaussian(X[:, columns], mi, sigma)
	plot_multi_gaussian(get_array(train, 2), get_array(train, 3), Y_multivariate_gaussian)

	#Parameters for Multivariate Gaussian for 3 features
	columns_f3 = [1,2]
	X = np.array(train)

	# Find parameters
	mi3 = get_mi(train, columns_f3)
	sigma3 = get_covariance_matrix(train, columns_f3)
	print "Multivariate Gaussian Parameters using 3 features:"
	print "Mi: ", mi3, "Sigma: ", sigma3
	print "\n"
	#F3 = multivariate_gaussian(X[:,columns_f3], mi3, sigma3)

	#Predicting VO2 Max
	A = np.array([[58.2, 302],[70.7, 530],[82, 208]])
	print "Predicted values for 2 feature Gaussiana: "
	y_A = multivariate_gaussian(A, mi3, sigma3)
	print y_A

	print "Predicted values using American College os Sports Equation:"
	a = american_college_sports_equation(A)
	print "\n"
	print a

	print "Erro Quadratico Medio"
	print mean_squared_error(y_A, a)

	test = np.array(test)
	columns_f4 = [0,1,2]
	mi4 = get_mi(train, columns_f4)
	sigma4 = get_covariance_matrix(train, columns_f4)
	print "Multivariate Gaussian Parameters using 3 features:"
	print mi4, sigma4 
	print "\n"

	y_f3 = multivariate_normal.pdf(test[:,columns_f3], mi3, sigma3)
	y_f4 = multivariate_normal.pdf(test[:,columns_f4], mi4, sigma4)

	print y_f3.shape, y_f4.shape, test[:,3].shape

	mse_f3 = mean_squared_error(test[:,3], y_f3)
	mse_f4 = mean_squared_error(test[:,3], y_f4)

	print "MSE for F3: ", mse_f3
	print "MSE for F4: ", mse_f4	

	# K-Means for 3 clusters
	K = 3
	k_means(data, K)

	#K-Means for 4 clusters
	K = 4
	k_means(data, K)

	# Question 3.1
	ids = split_ages(train)
	Y_pred_gaussian_mix = predict_multi_gaussian(train, ids)

	print "Mean Squared Errors for Each Model: "
	print "First model:"
	y_true = get_array(ids[0], 3)
	print mean_squared_error(np.squeeze(np.asarray(Y_pred_gaussian_mix[0])), y_true)
	print "\n"

	print "Second model:"
	y_true = get_array(ids[1], 3)
	print mean_squared_error(np.squeeze(np.asarray(Y_pred_gaussian_mix[1])), y_true)
	print "\n"

	print "Third model:"
	y_true = get_array(ids[2], 3)
	print mean_squared_error(np.squeeze(np.asarray(Y_pred_gaussian_mix[2])), y_true)
	print "\n"

	ids = ages_nb(ids)
	y_true = get_correct_classification(test)
	y_pred = naive_bayes(train, ids, test)

	print "Accuracy:"
	print len(y_true), len(y_pred)
	print accuracy_score(y_true, y_pred)

	#Mixture of Gaussians
	#y_true = get_correct_classification(test)
	y_true = get_array(test, 3)
	y_pred = mixture_gaussians(train, test, 3)

	print "Mean Squared Error"
	print mean_squared_error(y_true, y_pred)

main()
	