# itemcf_module.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

rating_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
item_sim = cosine_similarity(rating_matrix.T)

movie_ids = rating_matrix.columns

def get_itemcf_scores(user_id):
    user_ratings = rating_matrix.loc[user_id]
    scores = {}

    for idx, movie_id in enumerate(movie_ids):
        if user_ratings[movie_id] > 0:
            continue

        sim_sum, weighted = 0, 0
        for j, rated_movie in enumerate(movie_ids):
            if user_ratings[rated_movie] > 0:
                sim = item_sim[idx][j]
                weighted += sim * user_ratings[rated_movie]
                sim_sum += sim

        if sim_sum > 0:
            scores[movie_id] = weighted / sim_sum

    return scores