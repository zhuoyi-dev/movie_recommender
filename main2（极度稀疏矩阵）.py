import pandas as pd

# 读取数据
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

print("电影数量：", len(movies))
print("评分数量：", len(ratings))
print("用户数量：", ratings['userId'].nunique())

print("\nmovies 示例：")
print(movies.head())

print("\nratings 示例：")
print(ratings.head())