🎌 Anime Recommendation System

🧠 Project Description

This project is a Content-Based Anime Recommendation System that suggests similar anime titles based on user input.
It uses Natural Language Processing (NLP) techniques and cosine similarity to identify relationships between anime descriptions and recommend shows that share similar themes, genres, and narratives.

The goal of this project is to understand the core working principle behind modern recommendation engines (like those used by Netflix, Crunchyroll, and Spotify) using simple, interpretable, and effective data science techniques.

⸻

⚙️ Technologies & Libraries Used
	•	Python
	•	Pandas – for data cleaning and manipulation
	•	NumPy – for numerical operations
	•	Scikit-learn – for TF-IDF vectorization & cosine similarity
	•	Matplotlib / Seaborn (optional) – for visualization
	•	Jupyter Notebook – for experimentation and analysis

⸻

📂 Dataset
	•	Source: Kaggle - Anime Dataset
(You can mention the specific dataset name, e.g. Anime Recommendation Database 2020)
	•	The dataset contains details such as:
	•	Anime Name
	•	Genre
	•	Synopsis / Description
	•	Type (TV, Movie, OVA, etc.)
	•	Rating
	•	Members and Popularity

⸻

🧩 Project Workflow
	1.	Data Loading
Load the CSV file into a Pandas DataFrame.

import pandas as pd  
df = pd.read_csv("anime.csv")


	2.	Data Cleaning
	•	Removed missing values and duplicates
	•	Dropped unnecessary columns like anime_id
	•	Cleaned text fields for better NLP processing
	3.	Feature Extraction (TF-IDF Vectorization)
Used TfidfVectorizer to convert anime descriptions into numerical vectors for text analysis.

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])


	4.	Similarity Computation
Calculated cosine similarity to measure how close two anime are in terms of their textual features.

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


	5.	Recommendation Function
Created a function that returns the top 10 most similar anime based on cosine similarity scores.

indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def recommend_anime(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[anime_indices]


	6.	Example Output

recommend_anime("Naruto")

Output:

1. Naruto: Shippuden  
2. Bleach  
3. One Piece  
4. Fairy Tail  
5. Hunter x Hunter  
...



⸻

💡 Key Learnings
	•	Hands-on understanding of content-based recommendation systems
	•	Application of TF-IDF and cosine similarity for NLP tasks
	•	Insights into feature extraction, similarity computation, and filtering
	•	Experience in data preprocessing and pipeline design

⸻

🚀 Future Enhancements
	•	Add user-based collaborative filtering
	•	Include genre + synopsis hybrid model
	•	Deploy as a web app using Streamlit or Flask
	•	Integrate with an anime API for live recommendations

⸻

🧑‍💻 Author

Diwakar
Data Science Enthusiast | MCA Student | AI & ML Learner
📍 Passionate about turning data into intelligent, creative systems
