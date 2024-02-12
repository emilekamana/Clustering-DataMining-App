import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

st.set_page_config(page_title="News")

st.title('Select a demo on te left side nav')