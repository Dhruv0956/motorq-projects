# Cleo

Cleo is an AI-powered desktop assistant developed in Python that combines conversational AI, speech recognition, desktop automation, and third-party APIs in one application. It supports both text and voice workflows for productivity, information retrieval, entertainment, and utility tasks.

## Features

- Conversational assistant interface with Streamlit
- Voice input using SpeechRecognition
- Text-to-speech responses using pyttsx3
- Weather updates through the OpenWeatherMap API
- News retrieval through NewsAPI
- Spotify playback control through Spotipy and the Spotify Web API
- Desktop automation for launching common Windows apps and websites
- Productivity utilities such as to-do lists, reminders, and timers
- Internet speed testing
- Sudoku solving
- Optional MySQL connection helper for persistent integrations

## Technologies Used

- Python
- Streamlit
- SpeechRecognition
- pyttsx3
- Spotipy
- OpenWeatherMap API
- NewsAPI
- MySQL
- SQLite
- Requests
- speedtest-cli
- Git and GitHub

## Project Structure

```text
Cleo/
  app.py
  requirements.txt
  .env.example
  cleo/
    assistant.py
    automation.py
    config.py
    news.py
    speech.py
    spotify_client.py
    storage.py
    sudoku.py
    utils.py
    weather.py
  docs/
    Cleo-1.pdf
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add API keys where needed:

```powershell
copy .env.example .env
```

## Run

Start the Streamlit app:

```powershell
streamlit run app.py
```

## Optional API Setup

The app runs without API keys, but external services need environment variables:

```text
OPENWEATHER_API_KEY=your_openweathermap_key
NEWS_API_KEY=your_newsapi_key
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=cleo
```

## Notes

This repository version avoids hardcoded credentials. Features that depend on external services return a setup message until the relevant API keys are configured.
