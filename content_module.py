# content_module.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

movies['content'] = movies['title'] + ' ' + movies['genres']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

movie_id_to_index = pd.Series(movies.index, index=movies['movieId']).to_dict()
index_to_movie_id = pd.Series(movies['movieId'], index=movies.index).to_dict()

def build_profile(user_id):
    liked = ratings[(ratings['userId']==user_id)&(ratings['rating']>=4)]
    profile = np.zeros(tfidf_matrix.shape[1])

    for _,row in liked.iterrows():
        idx = movie_id_to_index[row['movieId']]
        profile += tfidf_matrix[idx].toarray().flatten()*row['rating']

    norm = np.linalg.norm(profile)
    if norm>0:
        profile/=norm
    return profile

def get_content_scores(user_id):
    profile = build_profile(user_id)
    scores = {}

    for idx in range(tfidf_matrix.shape[0]):
        movie_id = index_to_movie_id[idx]
        if movie_id in ratings[ratings['userId']==user_id]['movieId'].values:
            continue
        vec = tfidf_matrix[idx].toarray().flatten()
        scores[movie_id] = np.dot(profile, vec)

    return scores