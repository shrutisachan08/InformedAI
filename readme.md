## AI/ML News Application
This AI/ML-powered application is designed to provide a comprehensive news experience by fetching the latest news, detecting fake news, summarizing articles, and assisting users in improving the content. The app is built using Streamlit for an interactive and user-friendly interface.
Deployment link : https://informedai-ht5f2m3df3hdhftwksg9u8.streamlit.app/

Features

## 1.News Fetching:
The app fetches the latest news articles from various sources using the NewsAPI.
Users can browse through the news articles based on their preferences.

## 2.Fake News Detection:

The app uses machine learning models to analyze news articles and determine their authenticity.
It identifies whether the news is real or fake, helping users to stay informed with accurate information.

## 3.News Summarization:

The app provides concise summaries of news articles, allowing users to quickly grasp the key points without reading the entire article.

## 4.News Assistant:

The app assists users in improving the news content by providing suggestions and additional information.

## Installation
1.Clone the Repository - command : git clone https://github.com/your-username/your-repo-name.git
2.Install Dependencies: Install the required Python packages using the requirements.txt file.
command : pip install -r requirements.txt
3.Run the App: Start the Streamlit app
command : streamlit run main.py

## Usage
1.Navigate through different features of the app using the sidebar:
2.News Recommender: Fetches the latest news articles.
3.Fake News Detector: Analyzes the authenticity of the news.
4.Summarizer: Provides concise summaries of the news articles.
5.News Assistant: Assists in improving news content.

## API Key
The app requires an API key to fetch news articles. Replace the api_key in the code with your NewsAPI key.

## File Structure
1.main.py: The main entry point for the app, handling navigation between different features.
2.news_recommender.py: Contains the logic for fetching news articles.
3.fake.py: Contains the logic for detecting fake news.
4.summarizer.py: Contains the logic for summarizing news articles.
5.assistant.py: Contains the logic for the news assistant.

