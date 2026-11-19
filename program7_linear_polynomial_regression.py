# Program 7: Linear Regression (Boston Housing) and Polynomial Regression (Auto MPG)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ---------- Part 1: Linear Regression ----------
boston = fetch_openml(name="boston", version=1)
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["Target"] = boston.target

x = df[["RM"]]
y = df["Target"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
predict = LinearRegression().fit(xtrain, ytrain).predict(xtest)

plt.figure()
plt.scatter(xtest, ytest)
plt.plot(xtest, predict, c="red")
plt.xlabel("RM")
plt.ylabel("Target")
plt.title("Linear Regression")
plt.show()

# ---------- Part 2: Polynomial Regression ----------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)
data = data.dropna()

predict = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
).fit(data[["horsepower"]], data["mpg"]).predict(data[["horsepower"]])

plt.figure()
plt.scatter(data["horsepower"], data["mpg"])
sns.lineplot(x=data["horsepower"], y=predict, c="red")
plt.xlabel("horsepower")
plt.ylabel("mpg")
plt.title("Polynomial Regression (degree=2)")
plt.show()
