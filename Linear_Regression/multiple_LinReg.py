from audioop import add
from statistics import linear_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sb
from collections import Counter

from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

address = "C:\\Users\\Connor Wooley\\OneDrive\\Documents\\python\\Ex_Files_Python_Data_Science_EssT_Pt2\\Ex_Files_Python_Data_Science_EssT_Pt2\\Exercise Files\\Data\\enrollment_forecast.csv"

enroll = pd.read_csv(address)

enroll.columns = ['year', 'roll', 'unem', 'hgrad', 'inc']
# print(enroll.head())
# all vars need to be continuous numeric
# linear relationship between predictors and predictand

# print(sb.pairplot(enroll))
# plt.show()


# hgrad v unem:

# look at the relationship graphs to extrapolate a linear relationship against enroll
# check continuous 

# check for independence
# print(enroll.corr())

enroll_data = enroll[['unem', 'hgrad']].values
enroll_target = enroll[['roll']].values
enroll_data_names = ['unem', 'hgrad']

# scale the predictors (x = data, y = target)

X, y = scale(enroll_data), (enroll_target)

missing_values = X==np.NAN
# print(X[missing_values == True])
# check for empty array

LinReg = LinearRegression(normalize = True)

LinReg.fit(X,y)
# print(LinReg.score(X,y)) #R sqrd, how well this correlation predicts college enrollment