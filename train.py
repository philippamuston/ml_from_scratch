import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00','#0000FF' ])
from KNN import KNN
from linear_regression import LinearRegression

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=200)
#plt.show()
#X = DATA, y= LABELS

clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(predictions)
print(list(y_test))

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

##################################################

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

plt.figure()
plt.scatter(X[:, 0], y, color='b', marker='o', s=20)
#plt.show()

reg = LinearRegression(learning_rate=0.01)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

def mean_scored_error(y_test , predictions):
    return np.mean((y_test - predictions)**2)


mse = mean_scored_error(y_test, predictions)

#default learning rate of 0.001 gives 783, a faster learning rate of 0.01 gives 305 which is much better!
print(mse)
