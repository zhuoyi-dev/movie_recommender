import pandas as pd

# 你需要把这三个函数从各自文件 copy 过来
from usercf_module import get_usercf_scores
from itemcf_module import get_itemcf_scores
from content_module import get_content_scores


def hybrid_recommend(user_id, top_n=10):
    usercf = get_usercf_scores(user_id)
    itemcf = get_itemcf_scores(user_id)
    content = get_content_scores(user_id)

    all_movies = set(usercf) | set(itemcf) | set(content)

    final_scores = {}

    for m in all_movies:
        u = usercf.get(m, 0)
        i = itemcf.get(m, 0)
        c = content.get(m, 0)

        final_scores[m] = 0.4*u + 0.4*i + 0.2*c

    top = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print("\n🔥 Hybrid 最终推荐：\n")
    for movie_id, score in top:
        print(movie_id, score)


if __name__ == "__main__":
    hybrid_recommend(1)