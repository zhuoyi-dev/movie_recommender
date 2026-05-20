import pandas as pd
import numpy as np

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# 用户-电影矩阵
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

user_movie_filled = user_movie_matrix.fillna(0)

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# === 第一步：找和用户1最像的10个人 ===
target_user_id = 1
target_vector = user_movie_filled.loc[target_user_id].values

similarities = {}

for user_id in user_movie_filled.index:
    if user_id == target_user_id:
        continue
    other_vector = user_movie_filled.loc[user_id].values
    sim = cosine_similarity(target_vector, other_vector)
    similarities[user_id] = sim

top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
top_user_ids = [uid for uid, _ in top_users]

# === 第二步：找这些人喜欢但用户1没看过的电影 ===
target_user_rated = user_movie_matrix.loc[target_user_id]
unseen_movies = target_user_rated[target_user_rated.isna()].index

movie_scores = {}

for movie_id in unseen_movies:
    score = 0
    sim_sum = 0
    for uid, sim in top_users:
        rating = user_movie_matrix.loc[uid, movie_id]
        if not np.isnan(rating):
            score += sim * rating
            sim_sum += sim
    if sim_sum > 0:
        movie_scores[movie_id] = score / sim_sum

# === 第三步：排序，取前10推荐 ===
recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n给用户1推荐的电影：\n")
for movie_id, score in recommended_movies:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"{title}  推荐得分：{score:.2f}")