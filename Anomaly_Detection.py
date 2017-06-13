

from __future__ import division
import os
import sys
import numpy as np
from model import GMM
import matplotlib.pyplot as plt

DATASET_ROOT = os.path.join(os.path.dirname(__file__),"data")


def visualiseFit(data, model):
	"""Plot the contour lines of a 2D multivariate Gaussian"""
	
 	slack = 5  	 # margins for plot
	ticks = 100  # granularity level for contour lines
	lin = np.linspace(np.amin(data), np.amax(data), ticks)
	
	# generate data points
	[X, Y] = np.meshgrid(lin, lin)

	# compute the probabilities associated with the generated data
	genData = np.stack((X.flatten(), Y.flatten()), axis=1)
	proba = model.multivariateGaussian(genData, model.mu, model.sigma2)
	
	# reshape the probabilities to match X & Y
	Z = np.reshape(proba, (ticks, ticks))

	# plot the contour lines
	plt.figure()
	plt.scatter(data[:, 0], data[:, 1])
	plt.contour(X, Y, Z, levels= 
		[1e-20, 1e-10, 1e-8, 1e-6, 1e-5, 0.001, 0.009, 0.05, 0.1])
	plt.xlabel("Latency (ms)")
	plt.ylabel("Throughput (mb/sec)")
	plt.axis([min(data[:, 0]) - slack, max(data[:, 0]) + slack, min(data[:, 1]) - slack, max(data[:, 0]) + slack])
	plt.title("2D Gaussian model")
	plt.show()



def main():

	if True:
		### Peform anomaly detection on a dataset of servers with 2 features ###
		# import the 2D data 
		X = np.loadtxt(DATASET_ROOT + "/" + str(training_data), delimiter=",", usecols=(0,1))	
		Xval = np.loadtxt(DATASET_ROOT + "/" + str(validation_data), delimiter=",", usecols=(0,1))
		yval = np.loadtxt(DATASET_ROOT + "/" + str(validation_data), delimiter=",", usecols=[2])

				# plot the data
		print('\nPlotting the data...')
		slack = 5  # margin for plots
	
		plt.scatter(X[:, 0], X[:, 1])
		plt.xlabel("Latency (ms)")
		plt.ylabel("Throughput (mb/sec)")
		plt.axis([min(X[:, 0]) - slack, max(X[:, 0]) + slack, min(X[:, 1]) - slack, max(X[:, 0]) + slack])
		plt.show()

		# fit the model to the data
		print('\nFitting the model...')

		model = GMM()
		[mean, sigma2] = model.estimateGaussian(X)
		p = model.multivariateGaussian(X, mean, sigma2)

		# plot the model
		visualiseFit(X, model)

		# cross validate the optimal threshold
		print('\nCross-validating the threshold')
		pval = model.multivariateGaussian(Xval, mean, sigma2)
		[epsilon, F1] = model.gridSearch(pval, yval)

		print("Best threshold found using cross-validation: %s" %epsilon)
		print("Best F1 score on validation data: %s" %F1)

		# find the anomalies
		anomalies = np.where(p < epsilon)

		#plot the anomalies that were detected 
		plt.scatter(X[:, 0], X[:, 1])
		plt.scatter(X[anomalies, 0], X[anomalies, 1], color='red', facecolors='None', marker='o', s=100)
		plt.xlabel("Latency (ms)")
		plt.ylabel("Throughput (mb/sec)")
		plt.title("Anomalies detected")
		plt.axis([min(X[:, 0]) - slack, max(X[:, 0]) + slack, min(X[:, 1]) - slack, max(X[:, 0]) + slack])
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
		model = GMM()
		[mean, sigma2] = model.estimateGaussian(X)
		p = model.multivariateGaussian(X, mean, sigma2)

		# cross validate the optimal threshold
		pval = model.multivariateGaussian(Xval, mean, sigma2)
		[epsilon, F1] = model.gridSearch(pval, yval)

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

