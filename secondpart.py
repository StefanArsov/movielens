# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:50:59 2019

@author: Stefan
"""
import pandas as pd
import numpy as np




dataset = pd.merge(pd.merge(movies, ratings),users)
# 20 movies with highest rating
hig=dataset[['title','genres','rating']].sort_values('rating', ascending=False).head(20)



from sklearn.metrics.pairwise import pairwise_distances
# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0

# predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred
from sklearn.metrics import mean_squared_error
from math import sqrt

#calculate RMSE
def rmse(pred, actual):
    # nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))

#predict ratings-user simularity
user_prediction = predict(train_data_matrix, user_correlation, type='user')
#normalize ratings in user_prediction
tq=user_prediction[:,2]
norm=5*(tq-min(tq))/(max(tq)-min(tq))
user_prediction[:,2]=norm
#normalize ratings in item_prediction
item_prediction = predict(train_data_matrix, item_correlation, type='item')
#normalize ratings in item_prediction
tq1=item_prediction[:,2]
norm1=5*(tq1-min(tq1))/(max(tq1)-min(tq1))
item_prediction[:,2]=norm1
#calculate error in test and train
rmse_test_user=rmse(user_prediction, test_data_matrix)
rmse_test_item=rmse(item_prediction, test_data_matrix)
rmse_train_user=rmse(user_prediction, train_data_matrix)
rmse_train_item=rmse(item_prediction, train_data_matrix)

