from pickle import TRUE
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')

address = "C:\\Users\\Connor Wooley\\OneDrive\\Documents\\python\\Ex_Files_Python_Data_Science_EssT_Pt2\\Ex_Files_Python_Data_Science_EssT_Pt2\\Exercise Files\\Data\\titanic-training-data.csv"


titanic_training = pd.read_csv(address)
titanic_training.columns = ['PassengerID', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# print(titanic_training.head())

##############################identify a predictor that might correlate to the survived classification#############################

# print(titanic_training.info())
# look at the info and see if we have some features that may be problems (less entries than the total instances)

# check if target variable (survived) is binary

# sb.countplot(x = 'Survived', data = titanic_training, palette = 'hls')
# plt.show()
# if the graph has only two options then its a binary paramter

############################## detect missing values#############################
# print(titanic_training.isnull().sum())
# print(titanic_training.describe())
# if a feature has a ton of missing values its ok to remove
# if a feature could be a good predictor and has some missing values we can normalize it 

##############################dropping features#############################
titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
# print(titanic_data.head())


# Treating the Missing varibales for the remaining relevant features

# get an overview of the datapoints of the distribution between to relevant features 

# sb.boxplot(x='Parch', y='Age', data=titanic_data, palette='hls')
# plt.show()

parch_groups = titanic_data.groupby(titanic_data['Parch'])
# print(parch_groups.mean()) #shows categories for the feature and the mean age per category 

# print(parch_groups.mean())
parch_groups.mean()

def age_approx(cols):
    Age = cols[0]
    Parch = cols[1]
    # loop through rows where age has a null value and based on the Parch category, fill it with the mean
    if pd.isnull(Age):
        if Parch == 0:
            return 32.179 # the mean age in the zero parch category
        elif Parch == 1:
            return 24.422
        elif Parch == 2:
            return 17.217
        elif Parch == 3:
            return 33.2
        elif Parch == 4:
            return 44.5
        elif Parch == 5:
            return 39.2
        elif Parch == 6:
            return 43
        else:
            return 30
    else: 
        return Age

titanic_data['Age'] = titanic_data[['Age','Parch']].apply(age_approx, axis=1)
# print(titanic_data.isnull().sum())

titanic_data.dropna(inplace=True)
titanic_data.reset_index(inplace=True, drop=True)

# print(titanic_data.info())

############################## Converting categorical vars into a dummy indicator#############################


label_encoder = LabelEncoder()
gender_cat = titanic_data['Sex'] # encode so its binary

gender_encoded = label_encoder.fit_transform(gender_cat)
# print(gender_encoded[0:5])
# differentiate what 1 and 0 indicates

# print(titanic_data.head())
gender_df = pd.DataFrame(gender_encoded, columns = ['male_gender'])
# print(gender_df.head())

# encoding embarked var

embarked_cat = titanic_data['Embarked']
embarked_encoded = label_encoder.fit_transform(embarked_cat)
# print(embarked_encoded[:100])
# print(titanic_data)
# S-> 2, C->0, Q->1
# this is still not binary so we need to create a category that defines the port of embarkation so that feature can be binary
#One Hot Encoder

binary_encoder = OneHotEncoder(categories = 'auto')
embarked_1hot = binary_encoder.fit_transform(embarked_encoded.reshape(-1,1)) # outputting a column array
embarked_1hot_mat = embarked_1hot.toarray()
embarked_df =pd.DataFrame(embarked_1hot_mat, columns = ['C', 'Q', 'S'])

# print(embarked_df.head())

# now we have the sex and embarked df's, need to drop the sex and embarked data from the original df

titanic_data.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
# print(titanic_data.head())

# concatenate the two df's we created into the original df

titanic_dmy = pd.concat([titanic_data, gender_df, embarked_df], axis =1, verify_integrity = True,).astype(float)

# print(titanic_dmy.head())

########################### once in the correct format, check for independence#############################
# use the heatmap function from seaborn

# sb.heatmap(titanic_dmy.corr())
# plt.show()

# correlation close to 1 is strong and the assumption is a non correlation
# form heat map look at the white or black squares

titanic_dmy.drop(['Fare', 'Pclass'], axis=1, inplace=True)
# print(titanic_dmy.info())


######################### check if data set is of sufficient size#######################################
########################## have at least 50 records per perdictive features#############################


######################### break data into training and test sets########################################
X_train, X_test, y_train, y_test = train_test_split(titanic_dmy.drop(['Survived'], axis=1),
                                                    titanic_dmy['Survived'], test_size=0.2,
                                                    random_state=200)
# dont include the predictand variable
# print(X_train.shape)
# print(y_train.shape) # only want one column for our Y var

# print(X_train[0:5])


#########################Deploying and Evaluating the model##############################################
LogReg = LogisticRegression(solver = 'liblinear')
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
# print(y_pred)

#Evaluate with classification report

classification_report(y_test, y_pred)

# k-fold cross validation

ytrain_pred = cross_val_predict(LogReg, X_train, y_train, cv=5)
# print(confusion_matrix(y_train, ytrain_pred)) 
#-->> [[377  63],[ 91 180]]; 337 and 180 correct prediciton and 63, 91 incorrect predictions
# print(precision_score(y_train, ytrain_pred))

# make test prediction
titanic_dmy[863:864]
test_passenger = np.array([866,40,0,0,0,0,0,1]).reshape(1,-1)

# make a prediction
print(LogReg.predict(test_passenger))
print(LogReg.predict_proba(test_passenger))