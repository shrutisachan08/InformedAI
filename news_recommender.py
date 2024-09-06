# news_fetcher.py

import streamlit as st
from newsapi import NewsApiClient

class NewsFetcher:
    def __init__(self, api_key):
        self.newsapi = NewsApiClient(api_key=api_key)

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

    def run(self):
        st.title("News Fetcher")
        query = st.text_input("Enter your search query:")
        language = st.selectbox("Select language:", ['en', 'es', 'fr', 'de', 'it'])
        page_size = st.slider("Number of articles to fetch:", min_value=1, max_value=100, value=10)

        if st.button("Fetch News"):
            if query:
                articles = self.fetch_news(query=query, language=language, page_size=page_size)
                if articles:
                    for article in articles:
                        st.markdown(f"[{article['title']}]({article['url']})")
                        st.write(f"Description: {article['description']}")
                        st.write("-" * 80)
                else:
                    st.write("No articles found.")
            else:
                st.warning("Please enter a search query.")
if __name__ == "__main__":
    api_key = 'b8a03c825ab64da3bbf460c920f0b492'
    news_fetcher = NewsFetcher(api_key)
    news_fetcher.run()
