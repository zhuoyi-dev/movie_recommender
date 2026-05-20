import pandas as pd
import numpy as np

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# 计算每个用户的平均评分（消除偏置关键）
user_means = user_movie_matrix.mean(axis=1)

# 中心化评分矩阵
user_movie_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

def cosine_similarity(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    return np.dot(u, v) / (norm_u * norm_v)


target_user_id = 1
target_vector = user_movie_centered.loc[target_user_id].values

similarities = {}

for user_id in user_movie_centered.index:
    if user_id == target_user_id:
        continue
    other_vector = user_movie_centered.loc[user_id].values
    sim = cosine_similarity(target_vector, other_vector)
    similarities[user_id] = sim

top_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]

target_user_rated = user_movie_matrix.loc[target_user_id]
unseen_movies = target_user_rated[target_user_rated.isna()].index

movie_scores = {}

for movie_id in unseen_movies:
    score = 0
    sim_sum = 0
    for uid, sim in top_users:
        rating = user_movie_matrix.loc[uid, movie_id]
        if not np.isnan(rating):
            # 用“去偏置后的评分”
            score += sim * (rating - user_means[uid])
            sim_sum += abs(sim)
    if sim_sum > 0:
        # 最后加回用户自己的平均分
        movie_scores[movie_id] = user_means[target_user_id] + score / sim_sum

recommended_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("\n【改进后】给用户1推荐的电影：\n")
for movie_id, score in recommended_movies:
    title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f"{title}  预测评分：{score:.2f}")


# 电影的全局平均分
movie_mean = ratings.groupby('movieId')['rating'].mean()

if sim_sum > 0:
    pred_score = user_means[target_user_id] + score / sim_sum
else:
    # 冷门电影兜底策略
    pred_score = movie_mean.get(movie_id, 3.5)

# 裁剪范围
pred_score = min(5.0, max(0.5, pred_score))
movie_scores[movie_id] = pred_score