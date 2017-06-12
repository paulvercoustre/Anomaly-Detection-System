"""
A simple anomaly detection system using a mixture of Gaussians

The model estimates the means and variances of each feature
and fits a multivariate isotropic Gaussian to the data using these parameters

The probability threshold that minimises the F1 score on
the validation set is used as the decision boundary

"""

from __future__ import division
import os
import sys
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

DATASET_ROOT = os.path.join(os.path.dirname(__file__),"data")


def estimateGaussian(data):
	"""Estimate the means and variances of the data"""

	mu = np.mean(data, axis=0)
	print("The features means are: %s" %mu)
	
	sigma2 = np.var(data, axis=0)
	print("The features variances are: %s" %sigma2)

	return mu, sigma2


def multivariateGaussian(data, mu, cov):
	"""Compute the probability density function of data under
	the multivariate Gaussian distribution with params mu and cov"""

	if cov.ndim == 1:
		cov = np.diag(cov)

	p = multivariate_normal.pdf(data, mean=mu, cov=cov)

	return p


def gridSearch(proba, labels):
	"""Find the threshold that minimises the F1 score"""

	bestEpsilon = 0
	bestF1 = 0
	F1 = 0
	stepSize = (max(proba) - min(proba)) / 1000

	for epsilon in np.arange(min(proba), max(proba), stepSize):
		pred = proba < epsilon

		tp = sum((pred == 1) & (labels == 1))  # true positives
		fp = sum((pred == 1) & (labels == 0))  # false positives
		fn = sum((pred == 0) & (labels == 1))  # false negatives

		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		F1 = (2 * precision * recall) / (precision + recall)

		if F1 > bestF1:
			bestF1 = F1
			bestEpsilon = epsilon

	return bestEpsilon, bestF1


def visualiseFit(data, mu, cov):
	"""Plot the contour lines of a 2D multivariate Gaussian"""
	
	ticks = 100  # granularity level for contour lines
	lin = np.linspace(0, 30, ticks)
	
	# generate data points
	[X, Y] = np.meshgrid(lin, lin)

	# compute the probabilities associated with the generated data
	genData = np.stack((X.flatten(), Y.flatten()), axis=1)
	proba = multivariateGaussian(genData, mu, cov)
	
	# reshape the probabilities to match X & Y
	Z = np.reshape(proba, (ticks, ticks))

	# plot the contour lines
	plt.figure()
	plt.scatter(data[:, 0], data[:, 1])
	plt.contour(X, Y, Z, levels= 
		[1e-20, 1e-10, 1e-8, 1e-6, 1e-5, 0.001, 0.009, 0.05, 0.1])
	plt.xlabel("Latency (ms)")
	plt.ylabel("Throughput (mb/sec)")
	plt.axis([0, 30, 0, 30])
	plt.title("2D Gaussian model")
	plt.show()



def main():

	if False:

		### Peform anomaly detection on a dataset of servers with 2 features ###
		# import the 2D data 
		X = np.loadtxt(DATASET_ROOT + "/" + str(training_data), delimiter=",", usecols=(0,1))	
		Xval = np.loadtxt(DATASET_ROOT + "/" + str(validation_data), delimiter=",", usecols=(0,1))
		yval = np.loadtxt(DATASET_ROOT + "/" + str(validation_data), delimiter=",", usecols=[2])

		# plot the data
		print('\nPlotting the data...')
	
		plt.scatter(X[:, 0], X[:, 1])
		plt.xlabel("Latency (ms)")
		plt.ylabel("Throughput (mb/sec)")
		plt.axis([0, 30, 0, 30])
		plt.show()

		# fit the model to the data
		print('\nFitting the model...')
	
		[mean, sigma2] = estimateGaussian(X)
		p = multivariateGaussian(X, mean, sigma2)

		# plot the model
		visualiseFit(X, mean, sigma2)

		# cross validate the optimal threshold
		print('\nCross-validating the threshold')
		pval = multivariateGaussian(Xval, mean, sigma2)
		[epsilon, F1] = gridSearch(pval, yval)

		print("Best threshold found using cross-validation: %s" %epsilon)
		print("Best F1 score on validation data: %s" %F1)

		# find the anomalies
		anomalies = np.where(p < epsilon)

		#plot the anomalies that were detected 
		plt.scatter(X[:, 0], X[:, 1])
		plt.scatter(X[anomalies, 0], X[anomalies, 1], color='red')
		plt.xlabel("Latency (ms)")
		plt.ylabel("Throughput (mb/sec)")
		plt.title("Anomalies detected")
		plt.axis([0, 30, 0, 30])
		plt.show()


	if True:

		### Perform the same analysis, using an 11 dimensions dataset ###
		# import the 11D data 
		print('\nImporting 11D data')
		X = np.loadtxt(DATASET_ROOT + "/" + str(training_data2), delimiter=",", usecols=range(11))	
		Xval = np.loadtxt(DATASET_ROOT + "/" + str(validation_data2), delimiter=",", usecols=range(11))
		yval = np.loadtxt(DATASET_ROOT + "/" + str(validation_data2), delimiter=",", usecols=[11])

		# fit the model to the data
		print('\nFitting the model...')
		[mean, sigma2] = estimateGaussian(X)
		p = multivariateGaussian(X, mean, sigma2)

		# cross validate the optimal threshold
		pval = multivariateGaussian(Xval, mean, sigma2)
		[epsilon, F1] = gridSearch(pval, yval)

		print("Best threshold found using cross-validation: %s" %epsilon)
		print("Best F1 score on validation data: %s" %F1)

		# find the anomalies
		anomalies = np.where(p < epsilon)
		nb_outliers = len(anomalies[0])
		print("%s anomalies found" %nb_outliers)


if __name__ == "__main__":
	training_data = "X_2features.csv"
	validation_data = "Xval_2features.csv"
	training_data2 = "X_11features.csv"
	validation_data2 = "Xval_11features.csv"
	
	main()

