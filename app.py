from poster import get_poster

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = int(request.form["user_id"])
    results = hybrid(user_id, top_n=10, return_dict=True)

    movies = []
    for name, score in results.items():
        movies.append({
            "title": name,
            "score": round(score, 3),
            "poster": get_poster(name)
        })

    return render_template("recommend.html", movies=movies, user_id=user_id)