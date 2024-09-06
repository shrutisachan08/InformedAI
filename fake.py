# news_analyzer.py

import streamlit as st
from newsapi import NewsApiClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np
import random
import pandas as pd
import altair as alt

class NewsAnalyzer:
    def __init__(self, api_key):
        # Initialize the NewsApiClient with your API key
        self.newsapi = NewsApiClient(api_key=api_key)
        
        # Initialize the TF-IDF vectorizer and the classifier
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
        self.fake_news_detector = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_model_trained = False
    
    def fetch_news(self, query, language='en', from_date=None, to_date=None, page_size=10):
        try:
            response = self.newsapi.get_everything(
                q=query,
                language=language,
                from_param=from_date,
                to=to_date,
                page_size=page_size
            )
            return response['articles']
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return []
    
    def generate_fake_news(self, real_articles):
        fake_articles = []
        for article in real_articles:
            fake_title = f" {article['title']}"
            fake_description = f"This is a synthetic fake news article. It claims that {article['description']}"
            fake_articles.append({'title': fake_title, 'description': fake_description})
        return fake_articles
    
    def train_fake_news_detector(self, X_train, y_train):
        self.fake_news_detector.fit(X_train, y_train)
        self.is_model_trained = True
    
    def detect_fake_news(self, content):
        if not self.is_model_trained:
            raise ValueError("The model has not been trained yet. Please call train_fake_news_detector first.")
        
        features = self.vectorizer.transform([content])
        prediction = self.fake_news_detector.predict(features)
        score = max(self.fake_news_detector.predict_proba(features)[0])
        label = "FAKE" if prediction[0] == 0 else "REAL"
        return label, score
    
    def run(self):
        st.title("News Analyzer with Fake News Detection")
        
        # Sidebar
        st.sidebar.header("Settings")
        query = st.sidebar.text_input("Enter your search query:")
        language = st.sidebar.selectbox("Select language:", ['en', 'es', 'fr', 'de', 'it'])
        page_size = st.sidebar.slider("Number of articles to fetch:", min_value=10, max_value=100, value=50)
        
        if st.sidebar.button("Fetch and Analyze News"):
            if query:
                with st.spinner('Fetching and analyzing news...'):
                    real_articles = self.fetch_news(query=query, language=language, page_size=page_size)
                    if real_articles:
                        # Generate synthetic fake news
                        fake_articles = self.generate_fake_news(real_articles)
                        
                        # Combine real and fake articles
                        all_articles = real_articles + fake_articles
                        random.shuffle(all_articles)
                        
                        # Prepare data
                        texts = [f"{article['title']}. {article['description']}" for article in all_articles]
                        labels = [1] * len(real_articles) + [0] * len(fake_articles)
                        
                        # Vectorize the text data
                        X = self.vectorizer.fit_transform(texts)
                        y = np.array(labels)
                        
                        # Perform cross-validation
                        cv_scores = cross_val_score(self.fake_news_detector, X, y, cv=5)
                        
                        # Train the model on the full dataset
                        self.train_fake_news_detector(X, y)
                        
                        # Display results
                        st.success("Analysis complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.header("Model Performance")
                            st.write(f"Mean CV accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
                            
                            # Create a DataFrame for the CV scores
                            cv_df = pd.DataFrame({'Fold': range(1, 6), 'Accuracy': cv_scores})
                            
                            # Create an Altair chart
                            chart = alt.Chart(cv_df).mark_bar().encode(
                                x='Fold:O',
                                y='Accuracy:Q'
                            ).properties(
                                width=300,
                                height=200
                            )
                            st.altair_chart(chart)
                        
                        with col2:
                            st.header("Article Analysis")
                            results = []
                            for article in all_articles:
                                content = f"{article['title']}. {article['description']}"
                                label, score = self.detect_fake_news(content)
                                results.append({'Title': article['title'], 'Label': label, 'Confidence': score})
                            
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df)
                        
                        # Detailed article view
                        st.header("Detailed Article View")
                        for article in all_articles:
                            with st.expander(article['title']):
                                st.write(f"Description: {article['description']}")
                                content = f"{article['title']}. {article['description']}"
                                label, score = self.detect_fake_news(content)
                                st.write(f"Fake News Detection: {label} (Confidence: {score:.2f})")
                                if 'url' in article:
                                    st.write(f"[Read full article]({article['url']})")
                    else:
                        st.warning("No articles found.")
            else:
                st.sidebar.warning("Please enter a search query.")
        else:
         st.info("Enter a search query and click 'Fetch and Analyze News' to start.")
if __name__ == "__main__":
    api_key = 'b8a03c825ab64da3bbf460c920f0b492'
    news_analysis = NewsAnalyzer(api_key)
    news_analysis.run()
