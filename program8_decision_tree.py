# Program 8: Decision Tree on Breast Cancer dataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

cancer = load_breast_cancer()

xtrain, xtest, ytrain, ytest = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(criterion="gini").fit(xtrain, ytrain)
print("Test accuracy:", tree.score(xtest, ytest))

plt.figure(figsize=(20, 25))
plot_tree(tree, filled=True)
plt.show()
