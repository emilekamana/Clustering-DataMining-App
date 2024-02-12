import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

st.set_page_config(page_title="News")

st.title('Top headlines in 6 clusters based on descrtiptions')

API_KEY = os.getenv('API_KEY')

LOKY_MAX_CPU_COUNT = 5

# API endpoint to fetch news articles
news_api_url = "https://newsapi.org/v2/top-headlines?country=us&apiKey="+API_KEY

# Function to fetch articles and cache the response
def fetch_news():
    response = requests.get(news_api_url)
    print(response)
    articles = []
    if response.status_code == 200:
        articles = response.json().get("articles", [])
    else:
        print("error couldn't fetch")
    return articles

if 'articles' not in st.session_state:
    try:
        articles = fetch_news()
        st.session_state['articles'] = articles
    except:
        print("error couldn't fetch")
else:
    articles = st.session_state['articles']

print('articles:', len(articles))

texts = []
for article in articles:
    text = article.get("description") or article.get("content")
    if text:
        texts.append(text)

print('List: ',len(texts))

# Number of clusters
n_clusters = 5

# Cluster articles
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

# Display clusters
for i in range(n_clusters):
    st.header(f"Cluster {i+1}")
    cluster_articles = [articles[j] for j, label in enumerate(kmeans.labels_) if label == i]
    for article in cluster_articles:
        st.markdown(f"* [{article['title']}]({article['url']})")
