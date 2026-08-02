import streamlit as st

from cleo.assistant import CleoAssistant
from cleo.speech import listen_once, speak


st.set_page_config(page_title="Cleo", page_icon="C", layout="wide")

if "assistant" not in st.session_state:
    st.session_state.assistant = CleoAssistant()
if "history" not in st.session_state:
    st.session_state.history = []

assistant = st.session_state.assistant

st.title("Cleo")
st.caption("AI-powered desktop assistant for voice commands, productivity, automation, news, weather, Spotify, and utilities.")

with st.sidebar:
    st.header("Voice")
    use_voice = st.button("Listen")
    speak_response = st.checkbox("Speak responses", value=False)
    st.divider()
    st.header("Quick Commands")
    quick_commands = [
        "weather in Chennai",
        "latest news",
        "speed test",
        "open notepad",
        "open youtube",
        "solve sudoku",
    ]
    selected = st.selectbox("Try one", quick_commands)
    use_quick = st.button("Run command")

command = st.chat_input("Ask Cleo something...")

if use_voice:
    with st.spinner("Listening..."):
        command = listen_once()

if use_quick:
    command = selected

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.write(message)

if command:
    st.session_state.history.append(("user", command))
    with st.chat_message("user"):
        st.write(command)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = assistant.handle(command)
        st.write(response)

    st.session_state.history.append(("assistant", response))
    if speak_response:
        speak(response)
