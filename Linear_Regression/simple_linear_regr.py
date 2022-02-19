from statistics import linear_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale


rcParams['figure.figsize'] = 10,8

rooms =- np.random.rand(100,1)+3

# print(rooms)

price = 265 + 6*rooms + abs(np.random.randn(100,1))

# print(price)

# plt.plot(rooms, price, 'r^')
# plt.xlabel("# of rooms, 2019 avg")
# plt.ylabel("2019 avg num price in 1k USD")
# plt.show()

# simple linear regression

X = rooms
y = price

LinReg = LinearRegression()
LinReg.fit(X,y)
# print(LinReg.intercept_, LinReg.coef_)

print(LinReg.score(X,y)) # the R squared value

