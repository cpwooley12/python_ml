import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

plt.figure(figsize=(7,4))

iris = datasets.load_iris()

X = scale(iris.data)

y = pd.DataFrame(iris.target)

variable_names = iris.feature_names
# print(X[0:10])

############cluster this data#################
#lables predicted by the model
clustering = KMeans(n_clusters=3, random_state=5)

# fit model to data
clustering.fit(X)

iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_Length', 'Speal_Width', 'Petal_Length', 'Petal_Width']

y.columns = ['Targets']

color_theme = np.array(['darkgray', 'lightsalmon', 'powderblue'])

#subplot that has one row and two columns

plt.subplot(1,2,1)

plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c=color_theme[iris.target], s = 50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)

# show predicted values of y
plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c=color_theme[clustering.labels_], s = 50)
plt.title('K-Means Classification')

# plt.show()# predicted the clusters well but the labeling is off

# relabeling
 
relabel = np.choose(clustering.labels_, [2, 0, 1]).astype(np.int64)

plt.subplot(1,2,1)

plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c=color_theme[iris.target], s = 50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)

# show predicted values of y
plt.scatter(x = iris_df.Petal_Length, y = iris_df.Petal_Width, c=color_theme[relabel], s = 50)
plt.title('K-Means Classification')

# plt.show()
# compare the ground truth from the prediction made by the model

print(classification_report(y, relabel))
# precision is a measure of the model's relevancy
