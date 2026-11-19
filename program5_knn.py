# Program 5: k-Nearest Neighbors on 1D random data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x = np.random.rand(100).reshape(-1, 1)
y = np.array([1 if val <= 0.5 else 2 for val in x[:50]])

xtrain = x[:50]
ytrain = y
xtest = x[50:]

kValues = [1, 2, 3, 20, 40]

plt.figure(figsize=(20, 15))
for i, value in enumerate(kValues, 1):
    predicted = KNeighborsClassifier(n_neighbors=value).fit(xtrain, ytrain).predict(xtest)
    plt.subplot(5, 2, i)
    plt.scatter(xtrain, ytrain, c="red", label="Tested")
    plt.scatter(xtest, predicted, c="blue", label="Predicted")
    plt.title(f"k = {value}")
    plt.legend()
    plt.yticks(predicted)

plt.tight_layout()
plt.show()
