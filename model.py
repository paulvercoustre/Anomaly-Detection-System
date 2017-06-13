from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
	"""
	A simple anomaly detection system using a mixture of Gaussians

	The model estimates the means and variances of each feature
	and fits a multivariate isotropic Gaussian to the data using these parameters

	The probability threshold that minimises the F1 score on
	the validation set is used as the decision boundary

	"""

	def __init__(self):
		self.mu = []
		self.sigma2 = []
		self.bestEpsilon = 0
		self.bestF1 = 0

	def estimateGaussian(self, data):
		"""Estimate the means and variances of the data"""

		self.mu = np.mean(data, axis=0)
		print("The features means are: %s" %self.mu)
	
		self.sigma2 = np.var(data, axis=0)
		print("The features variances are: %s" %self.sigma2)

		return self.mu, self.sigma2

	def multivariateGaussian(self, data, mu, sigma2):
		"""Compute the probability density function of data under
		the multivariate Gaussian distribution with params mu and cov"""

		if sigma2.ndim == 1:
			cov = np.diag(sigma2)

		p = multivariate_normal.pdf(data, mean=mu, cov=cov)

		return p


	def gridSearch(self, proba, labels):
		"""Find the threshold that minimises the F1 score"""

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

			if F1 > self.bestF1:
				self.bestF1 = F1
				self.bestEpsilon = epsilon

		return self.bestEpsilon, self.bestF1

		