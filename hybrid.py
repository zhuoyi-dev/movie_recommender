from usercf_module import get_usercf_scores
from itemcf_module import get_itemcf_scores
from content_module import get_content_scores
import pandas as pd

movies = pd.read_csv('ml-latest-small/movies.csv')

def hybrid(user_id, top_n=10, return_dict=False):
    u = get_usercf_scores(user_id)
    i = get_itemcf_scores(user_id)
    c = get_content_scores(user_id)

    all_ids = set(u)|set(i)|set(c)
    final = {}

    for m in all_ids:
        final[m] = 0.4*u.get(m,0)+0.4*i.get(m,0)+0.2*c.get(m,0)

    top = sorted(final.items(), key=lambda x:x[1], reverse=True)[:top_n]

    print("\n🔥 Hybrid 最终推荐：\n")
    for movie_id,score in top:
        title = movies[movies['movieId']==movie_id]['title'].values[0]
        print(f"{title:<60} {score:.4f}")


    if return_dict:
        return {movies[movies['movieId']==mid]['title'].values[0]:score for mid,score in top}
if __name__=="__main__":
    hybrid(1)