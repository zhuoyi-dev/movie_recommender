import pandas as pd
import numpy as np

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# 用户-电影评分矩阵
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# 用 0 填充 NaN（非常关键）
user_movie_filled = user_movie_matrix.fillna(0)

# 余弦相似度函数
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# 计算 用户1 和 其他用户的相似度
user1 = user_movie_filled.loc[1].values

similarities = {}

for user_id in user_movie_filled.index:
    if user_id == 1:
        continue
    other_user = user_movie_filled.loc[user_id].values
    sim = cosine_similarity(user1, other_user)
    similarities[user_id] = sim

# 找最相似的前10个用户
top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]

print("和用户1最相似的前10个用户：")
for uid, sim in top_users:
    print(f"用户 {uid} 相似度：{sim:.4f}")