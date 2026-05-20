import pandas as pd
import numpy as np

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# 构建 电影-用户 矩阵（注意和之前反过来）
movie_user_matrix = ratings.pivot_table(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    return np.dot(u, v) / (norm_u * norm_v)

# 选一部电影，找和它最像的电影
target_movie_id = 1  # Toy Story
target_vector = movie_user_matrix.loc[target_movie_id].values

similarities = {}

for movie_id in movie_user_matrix.index:
    if movie_id == target_movie_id:
        continue
    other_vector = movie_user_matrix.loc[movie_id].values
    sim = cosine_similarity(target_vector, other_vector)
    similarities[movie_id] = sim

top_movies = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]

print("和 Toy Story 最相似的电影：\n")
for mid, sim in top_movies:
    title = movies[movies['movieId'] == mid]['title'].values[0]
    print(f"{title}  相似度：{sim:.4f}")