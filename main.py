from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    keywords = []
    explanation = ""
    rankings = []

    if request.method == "POST":
        job = request.form["job"]

        # Get multiple resumes
        resumes = request.form.getlist("resume")

        # Remove empty resumes
        resumes = [r for r in resumes if r.strip() != ""]

        if resumes and job.strip() != "":
            vectorizer = TfidfVectorizer(stop_words="english")

            documents = resumes + [job]
            tfidf_matrix = vectorizer.fit_transform(documents)

            job_vector = tfidf_matrix[-1]

            # Calculate similarity for each resume
            for i, resume in enumerate(resumes):
                resume_vector = tfidf_matrix[i]
                similarity = cosine_similarity(resume_vector, job_vector)
                percent = round(similarity[0][0] * 100, 2)
                rankings.append((f"Resume {i+1}", percent))

            # Sort by highest score
            rankings.sort(key=lambda x: x[1], reverse=True)

            score = rankings[0][1]

            # Extract top job keywords
            feature_names = vectorizer.get_feature_names_out()
            job_array = job_vector.toarray()[0]
            important_words = [
                feature_names[i]
                for i in job_array.argsort()[-5:]
            ]
            keywords = important_words[::-1]

            # Score explanation
            if score < 40:
                explanation = "Low alignment. Resume may be missing key required skills."
            elif score < 70:
                explanation = "Moderate alignment. Some required skills are present."
            else:
                explanation = "Strong alignment. Resume closely matches job requirements."

    return render_template(
        "index.html",
        score=score,
        keywords=keywords,
        explanation=explanation,
        rankings=rankings
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)