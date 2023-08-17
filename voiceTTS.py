import speech_recognition as sr
from gtts import gTTS
import os

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        audio = recognizer.listen(source)
        print(audio)
        
    try:
        text = recognizer.recognize_google(audio)
        return text
    
    except sr.UnknownValueError:
        print("Sorry, couldn't understand audio.")
        return "exit"
    
    except sr.RequestError as e:
        print("Error requesting results; {0}".format(e))
        return "exit"