import streamlit as st

# Import your different page modules
from news_recommender import NewsFetcher
from fake import NewsAnalyzer
from summarizer import run_app as run_summarizer
from assisstant import main as run_main

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [ "news_recommender", "fake","summarizer","assisstant" ])

# Navigation logic

if page == "news_recommender":
    api_key = 'b8a03c825ab64da3bbf460c920f0b492'
    news_fetcher = NewsFetcher(api_key)  # Create an instance of NewsFetcher
    news_fetcher.run()  # Call the run method on the instance


elif page == "fake":
    # app.py
    api_key = 'b8a03c825ab64da3bbf460c920f0b492'
    news_analyzer = NewsAnalyzer(api_key)
    news_analyzer.run()
elif page=="summarizer":
    run_summarizer()
elif page=="assisstant":
    run_main()