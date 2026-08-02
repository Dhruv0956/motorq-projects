import requests

from cleo.config import settings


def get_news(topic=None):
    if not settings.news_api_key:
        return "NewsAPI is not configured. Add NEWS_API_KEY to your .env file."

    endpoint = "https://newsapi.org/v2/everything" if topic else "https://newsapi.org/v2/top-headlines"
    params = {"apiKey": settings.news_api_key, "pageSize": 5, "language": "en"}
    if topic:
        params["q"] = topic
    else:
        params["country"] = "us"

    response = requests.get(endpoint, params=params, timeout=15)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    if not articles:
        return "I could not find news for that request."

    headlines = [article["title"] for article in articles[:5]]
    return "Top headlines:\n" + "\n".join(f"- {headline}" for headline in headlines)
