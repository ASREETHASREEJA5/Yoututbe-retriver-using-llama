import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os
import re
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

BASE_URL = 'https://www.googleapis.com/youtube/v3/search'
VIDEO_DETAILS_URL = 'https://www.googleapis.com/youtube/v3/videos'
FEEDBACK_FILE = 'feedback.json'  # File to store feedback

# Initialize session state for video links and feedback
if 'video_links' not in st.session_state:
    st.session_state.video_links = []
if 'feedback' not in st.session_state:
    st.session_state.feedback = []

# Initialize Groq API client
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile"
)

# Function to fetch YouTube video links and titles based on the question
def get_video_links_and_titles(query):
    params = {
        'part': 'snippet',
        'q': query,
        'key': API_KEY,
        'type': 'video',
        'maxResults': 10
    }
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        st.error(f"Error fetching videos for '{query}'. Status code: {response.status_code}")
        return []

    data = response.json()
    if 'items' not in data:
        st.warning(f"Error: 'items' key not found in the response for query '{query}'")
        return []

    video_data = []
    for item in data['items']:
        video_id = item['id']['videoId']
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        title = item['snippet']['title']
        video_data.append((title, video_url, video_id))

    return video_data

# Function to get video details (like count, comment count, etc.)
def get_video_details(video_ids):
    params = {
        'part': 'statistics,snippet',
        'id': ','.join(video_ids),
        'key': API_KEY
    }
    response = requests.get(VIDEO_DETAILS_URL, params=params)

    if response.status_code != 200:
        st.error(f"Error fetching video details. Status code: {response.status_code}")
        return []

    data = response.json()
    video_details = []
    for item in data['items']:
        video_id = item['id']
        stats = item.get('statistics', {})
        snippet = item.get('snippet', {})
        like_count = int(stats.get('likeCount', 0))
        comment_count = int(stats.get('commentCount', 0))
        published_date = snippet.get('publishedAt', "1970-01-01T00:00:00Z")
        published_years_ago = (datetime.now() - datetime.fromisoformat(published_date.replace("Z", ""))).days / 365
        video_details.append((video_id, like_count, comment_count, published_years_ago))

    return video_details

# Function to save feedback
def save_feedback(video_url, feedback):
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump([], f)

    with open(FEEDBACK_FILE, 'r') as f:
        feedback_data = json.load(f)

    feedback_data.append({'video_url': video_url, 'feedback': feedback})

    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_data, f)

# Function to retrieve feedback
def get_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

# Streamlit app
st.title("YouTube Video Search and Feedback")

# Query input
question = st.text_input("Enter your question:")
if st.button("Search"):
    if question:
        # Send the question to Groq model to get related search queries
        prompt = f"Provide the top 2 most relevant search queries for YouTube related to the given question: \"{question}\"."
        res = llm.invoke(prompt)

        raw_ques = res.content.split('\n')

        # Clean up and fetch video links and titles based on the Groq result
        ques = [re.sub(r'^\d+\.\s*', '', item).strip() for item in raw_ques if item.strip()]
        video_metadata = []
        video_ids = []
        for query in ques:
            videos = get_video_links_and_titles(query)
            video_metadata.extend(videos)
            video_ids.extend([video[2] for video in videos])  # Collect video IDs

        # Fetch additional details for all video IDs
        video_details = get_video_details(video_ids)
        video_details_dict = {vid[0]: vid[1:] for vid in video_details}

        # Rerank the videos based on similarity and other factors
        titles = [title for title, _, _ in video_metadata]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([question] + titles)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        ranked_videos = set()
        for (title, url, video_id), sim_score in zip(video_metadata, similarity_scores):
            like_count, comment_count, years_ago = video_details_dict.get(video_id, (0, 0, float('inf')))
            final_score = (
                sim_score * 0.6 +
                (like_count / 1000) * 0.2 +
                (comment_count / 100) * 0.1 +
                max(0, 1 - years_ago / 10) * 0.1
            )
            ranked_videos.add(((title, url), final_score))

        ranked_videos = list(ranked_videos)
        ranked_videos.sort(key=lambda x: x[1], reverse=True)

        top_3_videos = ranked_videos[:3]

        st.session_state.video_links = [(title, url) for (title, url), _ in top_3_videos]

    else:
        st.warning("Please enter a question to search.")

# Display video links and feedback buttons
if st.session_state.video_links:
    st.subheader("Top 3 Video Links:")
    for title, video_url in st.session_state.video_links:
        st.write(f"{title} - {video_url}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"👍 Like {video_url}", key=f"like-{video_url}"):
                save_feedback(video_url, "like")
                st.session_state.feedback.append({'video_url': video_url, 'feedback': 'like'})
        with col2:
            if st.button(f"👎 Dislike {video_url}", key=f"dislike-{video_url}"):
                save_feedback(video_url, "dislike")
                st.session_state.feedback.append({'video_url': video_url, 'feedback': 'dislike'})

# Feedback records
st.subheader("Feedback Records")
if st.session_state.feedback:
    for record in st.session_state.feedback:
        st.write(f"Video: {record['video_url']} - Feedback: {record['feedback']}")
else:
    st.write("No feedback records found.")
