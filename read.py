import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 读取数据
df_ratings = pd.read_csv("ratings.dat", sep="::", engine='python', names=["UserID", "MovieID", "Rating", "Timestamp"])
df_movies = pd.read_csv("movies.dat", sep="::", engine='python', names=["MovieID", "Title", "Genres"], encoding="ISO-8859-1")

df_ratings.drop(columns=["Timestamp"], inplace=True)  # 移除时间戳

# 2. 构建用户-电影评分矩阵
user_movie_matrix = df_ratings.pivot(index="UserID", columns="MovieID", values="Rating").fillna(0)

# 3. 计算用户相似度
user_sim_matrix = cosine_similarity(user_movie_matrix)
np.fill_diagonal(user_sim_matrix, 0)  # 自身相似度设为0

# 4. 生成推荐函数
def recommend_movies(user_id, k=10, top_n=10):
    if user_id not in user_movie_matrix.index:
        return []
    
    # 获取与当前用户最相似的K个用户
    sim_users = np.argsort(user_sim_matrix[user_id-1])[::-1][:k]  # 取前K个相似用户
    
    # 计算推荐评分
    user_ratings = user_movie_matrix.iloc[sim_users].mean(axis=0)  # 取K个用户的均值
    
    # 过滤掉当前用户已评分的电影
    seen_movies = user_movie_matrix.loc[user_id]
    user_ratings[seen_movies > 0] = 0
    
    # 取TOP N推荐
    top_movies = user_ratings.sort_values(ascending=False).head(top_n).index.tolist()
    return df_movies[df_movies.MovieID.isin(top_movies)][['MovieID', 'Title']]

# 5. 为用户1-20推荐电影
recommendations = {user_id: recommend_movies(user_id) for user_id in range(1, 21)}

# 6. 展示推荐结果
for user_id, movies in recommendations.items():
    print(f"用户 {user_id} 推荐电影:")
    print(movies)
    print("=" * 40)
