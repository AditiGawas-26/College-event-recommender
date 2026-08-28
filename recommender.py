"""
Core recommendation logic — shared by the terminal script (main.py) and
the web app (app.py) so both use the exact same matching code.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

events = [
    # ---- Workshops ----
    {"name": "AI & Machine Learning Workshop", "category": "Workshop",
     "tags": "AI, machine learning, coding, data science, neural networks"},
    {"name": "Web Dev Bootcamp", "category": "Workshop",
     "tags": "web development, coding, html, css, javascript, frontend"},
    {"name": "Cloud Computing 101", "category": "Workshop",
     "tags": "cloud, AWS, deployment, infrastructure, devops"},
    {"name": "UI/UX Design Sprint", "category": "Workshop",
     "tags": "design, figma, user experience, creativity, prototyping"},
    {"name": "Android App Dev Workshop", "category": "Workshop",
     "tags": "mobile development, android, kotlin, app design, coding"},
    {"name": "Cybersecurity Fundamentals", "category": "Workshop",
     "tags": "cybersecurity, hacking, networking, ethical hacking, security"},
    {"name": "Git & Open Source Workshop", "category": "Workshop",
     "tags": "git, github, open source, version control, collaboration"},
    {"name": "Blockchain Basics", "category": "Workshop",
     "tags": "blockchain, web3, cryptocurrency, smart contracts, coding"},
    {"name": "Resume & LinkedIn Clinic", "category": "Workshop",
     "tags": "career, resume, linkedin, personal branding, placements"},
    {"name": "Photography & Editing Workshop", "category": "Workshop",
     "tags": "photography, editing, lightroom, art, creativity"},
    {"name": "Digital Marketing Crash Course", "category": "Workshop",
     "tags": "marketing, social media, branding, analytics, business"},
    {"name": "Video Editing & Filmmaking Workshop", "category": "Workshop",
     "tags": "video editing, filmmaking, storytelling, premiere, creativity"},

    # ---- Hackathons ----
    {"name": "Code Sprint Hackathon", "category": "Hackathon",
     "tags": "coding, problem solving, teamwork, competition, hackathon"},
    {"name": "App Development Challenge", "category": "Hackathon",
     "tags": "mobile development, coding, app design, hackathon"},
    {"name": "Smart India Hackathon Prep Bootcamp", "category": "Hackathon",
     "tags": "SIH, hackathon, problem solving, prototyping, presentation, pitching"},
    {"name": "Game Jam Weekend", "category": "Hackathon",
     "tags": "game development, coding, design, creativity, teamwork"},
    {"name": "Fintech Hack", "category": "Hackathon",
     "tags": "fintech, coding, finance, innovation, teamwork"},
    {"name": "Sustainability Hackathon", "category": "Hackathon",
     "tags": "sustainability, environment, innovation, coding, social impact"},

    # ---- Competitions ----
    {"name": "Startup Pitch Night", "category": "Competition",
     "tags": "entrepreneurship, business, pitching, innovation, startup"},
    {"name": "Debate Championship", "category": "Competition",
     "tags": "debate, public speaking, critical thinking, argumentation"},
    {"name": "Cybersecurity CTF", "category": "Competition",
     "tags": "cybersecurity, hacking, coding, problem solving, capture the flag"},
    {"name": "Data Science Case Study Contest", "category": "Competition",
     "tags": "data science, analytics, machine learning, coding, case study"},
    {"name": "Business Case Study Challenge", "category": "Competition",
     "tags": "business, strategy, case study, consulting, presentation"},
    {"name": "Quiz Championship", "category": "Competition",
     "tags": "quiz, trivia, general knowledge, competition, teamwork"},
    {"name": "Chess Tournament", "category": "Competition",
     "tags": "chess, strategy, competition, focus, board games"},
    {"name": "Robo Race", "category": "Competition",
     "tags": "robotics, hardware, engineering, electronics, competition"},
    {"name": "Ideathon", "category": "Competition",
     "tags": "ideation, innovation, brainstorming, pitching, entrepreneurship"},

    # ---- Cultural ----
    {"name": "Battle of Bands", "category": "Cultural",
     "tags": "music, performance, band, competition, singing"},
    {"name": "Dance Fest", "category": "Cultural",
     "tags": "dance, performance, fitness, choreography, competition"},
    {"name": "Literature & Poetry Meet", "category": "Cultural",
     "tags": "writing, literature, poetry, creativity, spoken word"},
    {"name": "Drama & Theatre Night", "category": "Cultural",
     "tags": "drama, theatre, acting, performance, scriptwriting"},
    {"name": "Fashion Show", "category": "Cultural",
     "tags": "fashion, styling, ramp walk, performance, creativity"},
    {"name": "Stand-up Comedy Open Mic", "category": "Cultural",
     "tags": "comedy, standup, performance, humor, public speaking"},
    {"name": "Art & Sketching Exhibition", "category": "Cultural",
     "tags": "art, sketching, painting, creativity, exhibition"},
    {"name": "Photography Walk", "category": "Cultural",
     "tags": "photography, art, creativity, outdoors, exploration"},
    {"name": "Cultural Fusion Night", "category": "Cultural",
     "tags": "culture, tradition, music, dance, food, festival"},
    {"name": "Anime & K-drama Fan Meet", "category": "Cultural",
     "tags": "anime, k-drama, fan community, discussion, pop culture"},
    {"name": "Short Film Festival", "category": "Cultural",
     "tags": "film, filmmaking, storytelling, screening, creativity"},

    # ---- Sports ----
    {"name": "Cricket Tournament", "category": "Sports",
     "tags": "cricket, sports, teamwork, fitness, outdoor"},
    {"name": "Basketball League", "category": "Sports",
     "tags": "basketball, sports, fitness, teamwork, outdoor"},
    {"name": "Football Championship", "category": "Sports",
     "tags": "football, soccer, sports, teamwork, fitness"},
    {"name": "Badminton Open", "category": "Sports",
     "tags": "badminton, sports, fitness, competition, indoor"},
    {"name": "Table Tennis Tournament", "category": "Sports",
     "tags": "table tennis, sports, fitness, competition, indoor"},
    {"name": "Athletics Meet", "category": "Sports",
     "tags": "athletics, running, fitness, sports, track and field"},
    {"name": "Chess & Carrom Night", "category": "Sports",
     "tags": "chess, carrom, strategy, indoor games, competition"},
    {"name": "Yoga & Fitness Camp", "category": "Sports",
     "tags": "yoga, fitness, wellness, mindfulness, health"},
    {"name": "E-Sports Gaming Tournament", "category": "Sports",
     "tags": "esports, gaming, valorant, bgmi, competition, teamwork"},

    # ---- Exhibitions / Talks ----
    {"name": "Robotics Showcase", "category": "Exhibition",
     "tags": "robotics, hardware, engineering, electronics, innovation"},
    {"name": "Project Expo", "category": "Exhibition",
     "tags": "projects, engineering, innovation, showcase, presentation"},
    {"name": "Alumni Tech Talk", "category": "Seminar",
     "tags": "career, industry insights, networking, tech talk, mentorship"},
    {"name": "Public Speaking Masterclass", "category": "Seminar",
     "tags": "communication, public speaking, confidence, presentation"},
    {"name": "Women in Tech Panel", "category": "Seminar",
     "tags": "diversity, technology, career, panel discussion, networking"},
    {"name": "Career Fair", "category": "Seminar",
     "tags": "placements, internships, career, networking, recruiters"},

    # ---- Fest / Social ----
    {"name": "Annual Tech Fest", "category": "Fest",
     "tags": "technology, innovation, exhibitions, competitions, coding"},
    {"name": "Cultural Fest", "category": "Fest",
     "tags": "culture, music, dance, art, celebration, festival"},
    {"name": "Blood Donation Camp", "category": "Social",
     "tags": "volunteering, social service, health, community, donation"},
    {"name": "Campus Clean-up Drive", "category": "Social",
     "tags": "volunteering, sustainability, environment, community service"},
    {"name": "Coding Club Meetup", "category": "Social",
     "tags": "coding, community, networking, peer learning, club"},
]


CATEGORIES = sorted({event["category"] for event in events})


def get_recommendations(user_input, events=events, top_n=6, category=None):
    """
    Compute TF-IDF + cosine similarity between user_input and each event's
    tags, then return the top_n events (by relevance) with a match reason.
    Events with essentially zero similarity are excluded rather than padded in.

    If `category` is given (and isn't "All"), only events in that category
    are considered.
    """
    pool = [e for e in events if category in (None, "", "All") or e["category"] == category]
    if not pool:
        return []

    tag_texts = [event["tags"] for event in pool]
    all_texts = tag_texts + [user_input]

    # analyzer="char_wb" builds the vocabulary out of character n-grams
    # (within word boundaries) instead of whole words. This means a partial
    # word like "perform" still shares n-grams with "performance" (e.g.
    # "perfo", "rform"), so it gets nonzero similarity instead of being
    # treated as a totally unrelated token. It also gives some tolerance
    # for small typos, as a side benefit.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    user_vector = tfidf_matrix[-1]
    event_vectors = tfidf_matrix[:-1]

    similarity_scores = cosine_similarity(user_vector, event_vectors)[0]

    ranked_indices = np.argsort(similarity_scores)[::-1]
    min_score = 0.15
    relevant_indices = [idx for idx in ranked_indices if similarity_scores[idx] > min_score]
    top_indices = relevant_indices[:top_n]

    user_words = set(word.strip().lower() for word in user_input.split(","))

    max_score = float(similarity_scores[top_indices[0]]) if top_indices else 1.0

    results = []
    for idx in top_indices:
        event = pool[idx]
        score = float(similarity_scores[idx])

        event_words = set(tag.strip().lower() for tag in event["tags"].split(","))
        overlap = user_words & event_words
        reason = (f"matches your interest in {', '.join(overlap)}"
                  if overlap else "related to your interests")

        # Display strength is relative to the best match in this result set
        # (so the bar always reads meaningfully), while `score` keeps the
        # true cosine similarity for anyone who wants the raw number.
        strength = round(score * 100)

        results.append({
            "name": event["name"],
            "category": event["category"],
            "tags": [t.strip() for t in event["tags"].split(",")],
            "score": round(score, 2),
            "strength": strength,
            "reason": reason,
        })

    return results
