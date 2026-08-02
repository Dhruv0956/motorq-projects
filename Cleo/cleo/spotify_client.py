import spotipy
from spotipy.oauth2 import SpotifyOAuth

from cleo.config import settings


SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"


def _client():
    if not all([settings.spotify_client_id, settings.spotify_client_secret, settings.spotify_redirect_uri]):
        return None
    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
            scope=SCOPE,
        )
    )


def play(query=None):
    client = _client()
    if client is None:
        return "Spotify is not configured. Add Spotipy credentials to your .env file."
    if query:
        results = client.search(q=query, type="track", limit=1)
        tracks = results.get("tracks", {}).get("items", [])
        if not tracks:
            return f"I could not find {query} on Spotify."
        client.start_playback(uris=[tracks[0]["uri"]])
        return f"Playing {tracks[0]['name']} by {tracks[0]['artists'][0]['name']}."
    client.start_playback()
    return "Resuming Spotify playback."


def pause():
    client = _client()
    if client is None:
        return "Spotify is not configured. Add Spotipy credentials to your .env file."
    client.pause_playback()
    return "Paused Spotify playback."
