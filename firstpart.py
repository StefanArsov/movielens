# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:45:14 2019

@author: Stefan
"""
import pandas as pd
import numpy as np
file = open("users.dat","r") 
lines = file.readlines()
file.close()
lines = [elem.split('::') for elem in lines]
userID, genders, ages, occupations, zipcode = [], [], [], [], []
for elem in lines:
    userID.append(elem[0])
    genders.append(elem[1])
    ages.append(elem[2])
    occupations.append(elem[3])
    zipcode.append(elem[4])
users=pd.DataFrame({'user_id' : userID,'genders' : genders,'ages' : ages,'occupations' : occupations,'zipcode' : zipcode})
users["user_id"]=users["user_id"].astype('int64')
#users.set_index('userID',inplace=True)
file1 = open("movies.dat","r") 
lines1 = file1.readlines()
file1.close()
lines1 = [elem.split('::') for elem in lines1]
movieID,title, genres=[], [], []
for elem in lines1:
    movieID.append(elem[0])
    title.append(elem[1])
    genres.append(elem[2])
movies=pd.DataFrame({'movie_id':movieID,'title':title,'genres':genres})
movies["movie_id"]=movies["movie_id"].astype('int64')

movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

file2 = open("ratings.dat","r") 
lines2 = file2.readlines()
file2.close()
lines2 = [elem.split('::') for elem in lines2]
UserID, MovieID, Rating, Timestamp=[], [], [], []
for elem in lines2:
    UserID.append(elem[0])
    MovieID.append(elem[1])
    Rating.append(elem[2])
    Timestamp.append(elem[3])
ratings=pd.DataFrame({'user_id': UserID,'movie_id': MovieID,'rating': Rating})
ratings=ratings.astype('int64')
#'Timestamp': Timestamp
#ratings.merge(users, left_on='UserID', right_on='userID', how='outer')
#pomm1=pd.merge(movies,ratings)
#lens = pd.merge(pomm1, users)
#lens['Rating']=lens['Rating'].astype('int64')
#train=lens.head(350000)

#most_rated = lens.groupby('title').size().sort_values(ascending=False)[:30]
#total_ratings=lens.groupby('title')['Rating'].count().sort_values(ascending=False)


hr=genre_recommendations('Good Will Hunting (1997)').head(20)
ratings['user_id'] = ratings['user_id'].fillna(0)
ratings['movie_id'] = ratings['movie_id'].fillna(0)

# Replace NaN values in rating column with average of all values
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())
small_data = ratings.sample(frac=0.02)
print(small_data.info())
from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(small_data, test_size=0.25)
train_data_matrix = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
test_data_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
