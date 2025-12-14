from flask import Flask, render_template, request
import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# MongoDB Connection
# -----------------------------
MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client.netflix_db
movies_collection = db.movies
reco_collection = db.recommendations

# -----------------------------
# Load Movies from MongoDB
# -----------------------------
movies_cursor = movies_collection.find({}, {"_id": 0})
movies = pd.DataFrame(list(movies_cursor))
print('db connected')
print(movies.columns)

movies["genres"] = movies["genres"].fillna("")
movies_list = movies["title"].tolist()

# -----------------------------
# TF-IDF & Similarity
# -----------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_index = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def get_recommendations(movie_title, topN=10):
    movie_id = movie_index[movie_title]

    similarity_scores = list(enumerate(cosine_sim[movie_id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:topN+1]

    movie_indices = [i[0] for i in similarity_scores]
    scores = [i[1] for i in similarity_scores]

    return pd.DataFrame({
        "Movie": movies.iloc[movie_indices]["title"],
        "Similarity Score": scores
    })

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", movies_list=movies_list)

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_name = request.form["movie"]
    top_n = int(request.form["top_n"])

    recommendations = get_recommendations(movie_name, top_n)

    # Store recommendations in MongoDB
    reco_collection.delete_many({})
    reco_collection.insert_many(recommendations.to_dict("records"))

    return render_template(
        "data.html",
        Y="Recommendations generated successfully!",
        Z=recommendations.to_html(classes="table", index=False)
    )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
