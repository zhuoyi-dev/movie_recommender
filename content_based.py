import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取数据
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# ========= 1. 用 genres 做 TF-IDF 特征 =========
movies['content'] = movies['title'] + ' ' + movies['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

# 建立索引映射
movie_id_to_index = pd.Series(movies.index, index=movies['movieId']).to_dict()
index_to_movie_id = pd.Series(movies['movieId'], index=movies.index).to_dict()


def build_user_profile(user_id):
    """
    用 用户高评分电影 × 评分权重 构建用户画像向量
    """
    liked = ratings[
        (ratings['userId'] == user_id) & (ratings['rating'] >= 4.0)
    ][['movieId', 'rating']]

    profile = np.zeros(tfidf_matrix.shape[1])

    for _, row in liked.iterrows():
        movie_id = row['movieId']
        rating = row['rating']

        if movie_id not in movie_id_to_index:
            continue

        idx = movie_id_to_index[movie_id]
        movie_vec = tfidf_matrix[idx].toarray().flatten()

        # ⭐评分作为权重
        profile += movie_vec * rating

    # 归一化（非常关键）
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    return profile


def recommend_movies(user_id, top_n=10):
    user_profile = build_user_profile(user_id)

    scores = {}

    for idx in range(tfidf_matrix.shape[0]):
        movie_id = index_to_movie_id[idx]

        # 跳过用户看过的
        if movie_id in ratings[ratings['userId'] == user_id]['movieId'].values:
            continue

        movie_vec = tfidf_matrix[idx].toarray().flatten()

        score = np.dot(user_profile, movie_vec)
        scores[movie_id] = score

    # 排序
    sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\n🎬 基于内容（工业级）给用户 {user_id} 推荐：\n")
    for movie_id, score in sorted_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"{title:<60} 相似度：{score:.4f}")


if __name__ == "__main__":
    recommend_movies(user_id=1)