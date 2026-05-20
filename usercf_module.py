# usercf_module.py
import pandas as pd
import numpy as np

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

rating_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_means = rating_matrix.mean(axis=1)

def cosine(u, v):
    if np.linalg.norm(u)==0 or np.linalg.norm(v)==0:
        return 0
    return np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v))

def get_usercf_scores(user_id):
    target = rating_matrix.loc[user_id].fillna(0).values
    sims = {}

    for uid in rating_matrix.index:
        if uid == user_id:
            continue
        vec = rating_matrix.loc[uid].fillna(0).values
        sims[uid] = cosine(target, vec)

    top_users = sorted(sims.items(), key=lambda x:x[1], reverse=True)[:10]

    movie_scores = {}
    movie_mean = ratings.groupby('movieId')['rating'].mean()

    for movie_id in rating_matrix.columns:
        if not np.isnan(rating_matrix.loc[user_id, movie_id]):
            continue

        score, sim_sum = 0, 0
        for uid, sim in top_users:
            r = rating_matrix.loc[uid, movie_id]
            if not np.isnan(r):
                score += sim*(r-user_means[uid])
                sim_sum += sim

        if sim_sum > 0:
            pred = user_means[user_id] + score/sim_sum
        else:
            pred = movie_mean.get(movie_id, 3.5)

        movie_scores[movie_id] = pred

    return movie_scores