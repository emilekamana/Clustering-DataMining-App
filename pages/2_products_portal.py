import streamlit as st  
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None  # default='warn'

# Set page title
st.set_page_config(
    page_title="Products"
)
st.title("Nike's Products clustered ✔️")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('./data/nike_data_2022.csv')

# Extract relevant columns
relevant = df[["images", "name", "description"]]

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    # Remove stopwords and single character words
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Preprocess descriptions
relevant['preprocessed_description'] = relevant['description'].apply(preprocess_text)

# Feature extraction using CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform CountVectorizer
cv = count_vectorizer.fit_transform(relevant['preprocessed_description'])

n_clusters = 5

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=1)
clusters = kmeans.fit_predict(cv)

# Add cluster labels to the dataframe
relevant['cluster'] = clusters

for i in range(n_clusters):
    st.header(f"Cluster {i+1}")
    cluster_products = relevant[relevant['cluster'] == i]

    for _, product in cluster_products.iterrows():
        image_url = str(product['images']) 
        col1, col2 = st.columns(2)
        with col1:
            # Check if the image URL is valid
            if image_url.strip() and image_url.lower() != 'nan':  
                st.image(image_url.split(" | ")[0], use_column_width=True)
            else:
                st.write('Unable to retrieve image')
        with col2:
            st.subheader(product['name'])
            st.write(product['description'])
        st.markdown("---")

