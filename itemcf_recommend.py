import pandas as pd
import numpy as np

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# 电影-用户矩阵
movie_user_matrix = ratings.pivot_table(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)

# 用户-电影矩阵（判断看没看过用）
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    return np.dot(u, v) / (norm_u * norm_v)

# 预计算电影相似度（这里只算需要的）
def get_similar_movies(target_movie_id, top_n=20):
    target_vector = movie_user_matrix.loc[target_movie_id].values
    sims = {}
    for movie_id in movie_user_matrix.index:
        if movie_id == target_movie_id:
            continue
        other_vector = movie_user_matrix.loc[movie_id].values
        sims[movie_id] = cosine_similarity(target_vector, other_vector)
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]

# 给用户1推荐
target_user_id = 1
rated_movies = user_movie_matrix.loc[target_user_id].dropna()

scores = {}

for movie_id, rating in rated_movies.items():
    similar_movies = get_similar_movies(movie_id, top_n=20)
    for sim_movie_id, sim in similar_movies:
        if sim_movie_id in rated_movies.index:
            continue
        scores.setdefault(sim_movie_id, 0)
        scores[sim_movie_id] += sim * rating

# 排序推荐
recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("ItemCF 给用户1推荐：\n")
for mid, score in recommended:
    title = movies[movies['movieId'] == mid]['title'].values[0]
    print(f"{title}  推荐分：{score:.2f}")