# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats
from numpy import median
from sklearn import metrics
from sklearn import model_selection
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#RMSLE score calculation
def rmsle_score(y_true, y_pred):
    for i, y in enumerate(y_pred):
        if y_pred[i] < 0:
            y_pred[i] = 0
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

#Reading Datasets
filename = 'dataset/train.csv' 
df_train = pd.read_csv(filename)

df_train.shape
df_test = pd.read_csv('dataset/test.csv')

df_test.shape
df_train.isnull().sum()
df_train.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

#Setting training data
df_train = df_train.drop(['windspeed','atemp', 'casual', 'registered'], axis=1)
X = df_train[['temp', 'humidity', 'hour', 'month','workingday', 'holiday', 'season','weekday', 'year', 'weather']]
y = df_train['count']
# Training and test data is created by splitting the main data. 20% of test data is considered
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestRegressor(n_estimators=1000, max_depth=25, n_jobs = -1, random_state = 0)
kn = KNeighborsRegressor(n_neighbors=7,  weights='distance', p=1, n_jobs = -1)
ext = ExtraTreesRegressor(n_estimators=100,criterion='mse',max_depth=25, random_state = 42, n_jobs= -1)
bag = BaggingRegressor(n_estimators = 1000, random_state = 42)
hgb = HistGradientBoostingRegressor(max_iter = 1000, max_depth = 20, learning_rate = 0.1, random_state = 42)
xg = XGBRegressor(subsample = 0.5, max_depth= 9, nthread = -1, silent = 1)

voting = VotingRegressor(estimators=[('kn', kn), ('clf', clf), ('ext',ext), ('bag',bag), ('hgb', hgb),('xg',xg)], weights=[1,1,2,1,1,5], n_jobs=-1)
voting.fit(X_train,np.log(y_train))
y_pred_voting = np.exp(voting.predict(X_test))
print('Voting RMSLE score:', rmsle_score(y_test, y_pred_voting))

filename = 'dataset/test.csv' 
df_test = pd.read_csv(filename)

df_test.shape
df_test.rename(columns={'weathersit':'weather',
                     'mnth':'month',
                     'hr':'hour',
                     'yr':'year',
                     'hum': 'humidity',
                     'cnt':'count'},inplace=True)

df_test = df_test.drop(['windspeed','atemp'], axis=1)
df_test = df_test[['temp', 'humidity', 'hour', 'month', 'workingday', 'holiday', 'season','weekday', 'year', 'weather']]
df_test.shape

y_pred = np.exp(voting.predict(df_test))
True in (y_pred < 0)
for i, y  in enumerate(y_pred):
    if y_pred[i] < 0:
        y_pred[i] = 0

submission = pd.DataFrame()
submission['Id'] = range(y_pred.shape[0])
submission['Predicted'] = y_pred
submission.to_csv("new_submission.csv", index=False)

