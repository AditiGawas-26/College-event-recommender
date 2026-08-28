# College Event Recommender

A web-based College Event Recommendation System that helps students discover relevant college events based on their interests.

The system uses TF-IDF Vectorization and Cosine Similarity to compare a user's interests with event tags and recommend relevant events.

## Features

- Search for events based on interests
- Category-based event filtering
- Recommendations ranked by relevance
- Match strength display
- Explanation of why an event matches
- Flexible matching for related or partial terms
- Responsive web interface
- Handles cases where no relevant event is found

## Event Categories

- Workshop
- Hackathon
- Competition
- Cultural
- Sports
- Exhibition
- Seminar
- Fest
- Social

## Technologies Used

- Python
- Flask
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- HTML
- CSS

## How It Works

1. The user enters one or more interests.
2. The user can select an event category.
3. The system compares the input with tags associated with available events.
4. TF-IDF converts the text into numerical representations.
5. Cosine Similarity calculates relevance scores.
6. Relevant events are ranked and displayed.

## Project Structure

```text
college-event-recommender/
├── app.py
├── main.py
├── recommender.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Web Application

```bash
python app.py
```

Open the local URL shown in the terminal.

## Run the Terminal Version

```bash
python main.py
```

## Future Improvements

- Save favourite events
- Add event dates, venues, and timings
- Personalized user profiles
- Interest discovery quiz
- Database integration
