"""
College Event Recommender — web version.
Run with: python3 app.py, then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request
from recommender import events, get_recommendations, CATEGORIES

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    submitted = False
    user_input = ""
    selected_category = "All"

    if request.method == "POST":
        user_input = request.form.get("interests", "").strip()
        selected_category = request.form.get("category", "All") or "All"
        submitted = True
        if user_input:
            recommendations = get_recommendations(
                user_input, events, top_n=6, category=selected_category
            )

    return render_template(
        "index.html",
        recommendations=recommendations,
        submitted=submitted,
        user_input=user_input,
        categories=CATEGORIES,
        selected_category=selected_category,
        total_events=len(events),
    )


if __name__ == "__main__":
    app.run(debug=True)
