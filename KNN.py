import numpy as np
from collections import Counter

print("K Nearest Neighbours")
#for a datapoint calculate its distance from other datapoints in set. Get the closest K points, 
# Regression: Get the average of their values
# Classification: Get the label with the majority vote

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    #takes dataset as parameter X
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions


    def _predict(self, x):
        #compute the distnace
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        #get the indices for the closest neighbourds
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        #majority vote
        most_common = Counter(k_nearest_labels).most_common()
        #return highest voted/most common label 
        return most_common[0][0]