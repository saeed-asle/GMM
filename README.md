# Gaussian Mixture Model Implementation

Authored by Saeed Asle

Description
This project demonstrates a simple implementation of a Gaussian Mixture Model (GMM) in Python. The model is used to cluster a synthetic dataset generated from two distinct Gaussian distributions. The implementation includes functions for the Expectation and Maximization steps of the GMM algorithm.

# Features
The code provides the following features:

1.Data Generation: Generates synthetic data from two Gaussian distributions.

      Group 1: Mean = [-1, -1], Covariance = [[0.8, 0], [0, 0.8]]
      Group 2: Mean = [1, 1], Covariance = [[0.75, -0.2], [-0.2, 0.6]]
      
2.Visualization: Plots the generated data points and the final predicted clusters.

3.Expectation Step: Computes the responsibilities of each component for every data point.

4.Maximization Step: Updates the parameters (means, covariances, and weights) of the Gaussian components based on the computed responsibilities.

5.Convergence Check: Iterates the Expectation and Maximization steps until convergence or a maximum number of iterations is reached.

6.Prediction: Assigns data points to the component with the highest responsibility.

# Output

* Scatter plots of the initial data and the final predicted clusters.

* Console output of the final means, covariances, weights, and the number of iterations until convergence.
