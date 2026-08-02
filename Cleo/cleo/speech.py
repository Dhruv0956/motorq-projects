import speech_recognition as sr
import pyttsx3


def listen_once(timeout=5, phrase_time_limit=8):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "I could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
