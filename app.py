import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Insurance Review Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Insurance Review Sentiment Analysis")
st.markdown("""
This app analyzes sentiment in insurance reviews using different models.
""")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page", 
    ["Home", "Data Analysis", "Sentiment Prediction"]
)

# Load your trained models and vectorizers
@st.cache_resource
def load_models():
    # Load your models here
    with open('tfidf_model.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)
    with open('w2v_model.pkl', 'rb') as f:
        w2v_model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_model, w2v_model, tfidf_vectorizer

# Load your preprocessing functions
def preprocess_text(text):
    # Your preprocessing function here
    pass

# Home page
if page == "Home":
    st.header("Welcome to the Insurance Review Analyzer")
    st.write("""
    This application helps analyze insurance reviews using different models:
    - TF-IDF with SVM
    - Word2Vec with Logistic Regression
    - Transformer-based models
    """)
    
    # Add some example metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="TF-IDF Accuracy", value="85%")
    with col2:
        st.metric(label="Word2Vec Accuracy", value="83%")
    with col3:
        st.metric(label="BERT Accuracy", value="87%")

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    # Upload data option
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Show basic statistics
        st.subheader("Dataset Overview")
        st.write(f"Number of reviews: {len(df)}")
        
        # Visualizations
        st.subheader("Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig, ax = plt.subplots()
            df['note'].value_counts().plot(kind='bar')
            plt.title('Rating Distribution')
            st.pyplot(fig)
            
        with col2:
            # Average rating by product
            fig, ax = plt.subplots()
            df.groupby('produit')['note'].mean().plot(kind='bar')
            plt.title('Average Rating by Product')
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Sentiment Prediction page
elif page == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    
    # Load models
    try:
        tfidf_model, w2v_model, tfidf_vectorizer = load_models()
        
        # Text input
        user_text = st.text_area("Enter your review text:", height=100)
        
        if st.button("Analyze Sentiment"):
            if user_text:
                # Preprocess text
                processed_text = preprocess_text(user_text)
                
                # Make predictions with different models
                # TF-IDF prediction
                tfidf_vec = tfidf_vectorizer.transform([processed_text])
                tfidf_pred = tfidf_model.predict(tfidf_vec)[0]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("TF-IDF + SVM Prediction")
                    sentiment = ["Negative", "Neutral", "Positive"][tfidf_pred]
                    st.write(f"Sentiment: {sentiment}")
                
                with col2:
                    st.subheader("Word2Vec Prediction")
                    # Add your Word2Vec prediction here
                    
            else:
                st.warning("Please enter some text to analyze.")
                
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")

# Add footer
st.markdown("""
---
Lucas
""")