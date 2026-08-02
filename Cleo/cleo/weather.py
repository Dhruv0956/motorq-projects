import requests

from cleo.config import settings


def get_weather(city):
    if not settings.openweather_api_key:
        return "OpenWeatherMap is not configured. Add OPENWEATHER_API_KEY to your .env file."

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": settings.openweather_api_key,
            "units": "metric",
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    description = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    return f"The weather in {city} is {description}, {temp:.1f} C, feels like {feels_like:.1f} C."
