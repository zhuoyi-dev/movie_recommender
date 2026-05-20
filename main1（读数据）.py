import pandas as pd

movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# 构建 用户-电影 评分矩阵
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

print("评分矩阵形状：", user_movie_matrix.shape)
print(user_movie_matrix.head())