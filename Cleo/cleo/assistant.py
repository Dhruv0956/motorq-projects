import re

from cleo.automation import open_app, open_website
from cleo.news import get_news
from cleo.spotify_client import pause, play
from cleo.storage import add_todo, list_todos
from cleo.sudoku import demo_solution
from cleo.utils import run_speed_test
from cleo.weather import get_weather


class CleoAssistant:
    def handle(self, command):
        text = command.lower().strip()

        if text in {"hi", "hello", "hey"}:
            return "Hi, I am Cleo. How can I help?"

        if text.startswith("weather"):
            city = _after_keywords(command, ["weather in", "weather for", "weather"])
            return get_weather(city or "Chennai")

        if "news" in text:
            topic = _after_keywords(command, ["news about", "news on"])
            return get_news(topic)

        if text.startswith("open "):
            target = command[5:].strip()
            if target.lower() in {"google", "youtube", "wikipedia", "spotify"}:
                return open_website(target)
            return open_app(target)

        if text.startswith("search "):
            query = command[7:].strip()
            return open_website(query)

        if text.startswith("add todo "):
            return add_todo(command[9:].strip())

        if "todo" in text or "to-do" in text:
            return list_todos()

        if "speed" in text and "test" in text:
            return run_speed_test()

        if "sudoku" in text:
            return "Solved Sudoku:\n" + demo_solution()

        if text.startswith("play "):
            return play(command[5:].strip())

        if "pause spotify" in text or text == "pause":
            return pause()

        return (
            "I can help with weather, news, Spotify playback, opening apps, "
            "web searches, to-do lists, speed tests, and Sudoku solving."
        )


def _after_keywords(command, keywords):
    for keyword in keywords:
        match = re.search(re.escape(keyword), command, flags=re.IGNORECASE)
        if match:
            value = command[match.end():].strip(" :,-")
            if value:
                return value
    return ""
