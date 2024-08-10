from flask import Flask, request, render_template_string
from collections import Counter
import spacy

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the relevant words for document retrieval queries
relevant_words = ["report", "document", "paper", "file", "details", "information", "data", "statistics", "figures", "summary", "review"]

# Function to classify the query
def classify_query(query):
    # Tokenize the query
    query_tokens = query.lower().split()

    # Calculate term frequency (TF) for the query
    tf_scores = Counter(query_tokens)

    # Filter the TF scores for relevant words
    filtered_tf_scores = {word: tf_scores[word] for word in relevant_words}

    # Calculate the general score (sum of all relevant TF scores)
    tf_score = sum(filtered_tf_scores.values())

    # Calculate the length score (number of tokens)
    length_score = len(query_tokens)

    # Named Entity Recognition (NER)
    doc = nlp(query)
    ner_score = len(doc.ents)  # Number of named entities

    # Calculate the weighted average score
    avg_score = (0.5 * tf_score) + (0.3 * length_score) + (ner_score * 0.2)

    # Define a threshold for classification (can be tuned based on your data)
    threshold = 1.5

    # Classify the query based on the weighted average score
    if avg_score >= threshold:
        return "Document Retrieval Query"
    else:
        return "General Query"

# Define the HTML template
template = """
<!doctype html>
<html>
    <head>
        <title>Query Classifier</title>
    </head>
    <body>
        <h1>Query Classifier</h1>
        <form method="post" action="/">
            <label for="query">Enter your query:</label>
            <input type="text" id="query" name="query">
            <input type="submit" value="Classify">
        </form>
        {% if result %}
            <h2>Result:</h2>
            <p>Query: {{ query }}</p>
            <p>Classification: {{ result }}</p>
        {% endif %}
    </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def classify():
    result = None
    query = None
    if request.method == "POST":
        query = request.form["query"]
        result = classify_query(query)
    return render_template_string(template, result=result, query=query)

if __name__ == "__main__":
    app.run(debug=True)
