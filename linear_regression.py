#Find y=wx+b 
#w = weight, b = bias
#We want to find the best-fitting line through data with minimum error (calculate the derivative)
#Learning rate = how fast or slow to go in the direction of the Gradient Descent to get to minimum error. Too slow = never reach it, too high = bounce around and miss it

#Training: Initialise weight and bias as 0
#Predict result using y = wx + b
#Calculate errror
#Repeat n times
#X = [x1, x2, ... xn]
#with weight: wX = [wx1, wx2 ...wxn]
#with weight and bias: [wx1 + b, wx2 + b ...wxn + b]

#arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  = (2,4)

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.001, n_iterations = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations =  n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):

            #we're working with an array rather than one sample at a time, so work with dot products
            y_pred = np.dot(X, self.weights) + self.bias

            #Calculate gradient of error function in terms of weight, use the Transpose of X to get the correct dimensions
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))

            #Calculate bias
            db = (1/n_samples) * np.sum(y_pred - y)

            #Update the weight by the learning rate
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

