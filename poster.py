import requests

API_KEY = "你的TMDB_API_KEY"

def get_poster(movie_name):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": API_KEY,
        "query": movie_name
    }
    r = requests.get(url, params=params).json()

    if r["results"]:
        poster_path = r["results"][0]["poster_path"]
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path

    return "https://via.placeholder.com/300x450?text=No+Image"