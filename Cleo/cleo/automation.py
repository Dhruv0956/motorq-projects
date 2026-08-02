import os
import subprocess
import webbrowser


APP_COMMANDS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "command prompt": "cmd.exe",
}

WEB_COMMANDS = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "wikipedia": "https://www.wikipedia.org",
    "spotify": "https://open.spotify.com",
}


def open_app(name):
    normalized = name.lower().strip()
    command = APP_COMMANDS.get(normalized)
    if not command:
        return f"I do not have an app shortcut configured for {name}."

    subprocess.Popen(command, shell=False)
    return f"Opening {name}."


def open_website(name_or_url):
    target = name_or_url.lower().strip()
    url = WEB_COMMANDS.get(target, name_or_url)
    if not url.startswith(("http://", "https://")):
        url = f"https://www.google.com/search?q={target.replace(' ', '+')}"
    webbrowser.open(url)
    return f"Opening {url}."


def open_path(path):
    if not os.path.exists(path):
        return f"I could not find {path}."
    os.startfile(path)
    return f"Opening {path}."
