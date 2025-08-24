import os
import time
import uuid
import threading
import io
from gtts import gTTS
from dotenv import load_dotenv
from google import genai

load_dotenv()

class DetectionLog:
    def __init__(self, repeat_interval=5.0):
        """
        repeat_interval: Minimum seconds between repeating the same detection
        """
        self.last_spoken = {}  # {detection: timestamp}
        self.repeat_interval = repeat_interval
        self.history = []

        # Initialize Gemini client
        key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)

        # Local-only audio setup
        self.local_audio_supported = False
        try:
            import pygame
            pygame.mixer.init()
            self.local_audio_supported = True
        except Exception as e:
            print(f"[INFO] Local audio not supported: {e}")

    def log(self, detection, cloud_mode=False):
        """
        Log detection and speak it if enough time has passed.
        cloud_mode=True: Streamlit Cloud compatible
        """
        now = time.time()
        if detection not in self.last_spoken or (now - self.last_spoken[detection]) > self.repeat_interval:
            self.last_spoken[detection] = now

            # Ask Gemini for explanation
            explanation = self.ask_gemini(detection)

            # Speak
            if cloud_mode:
                # Streamlit Cloud: synchronous call
                self.speak(explanation, cloud_mode=True)
            else:
                # Local: background thread
                if self.local_audio_supported:
                    threading.Thread(target=self.speak, args=(explanation, False), daemon=True).start()

            return detection, explanation
        return None, None

    def speak(self, text, cloud_mode=False, lang="en"):
        """
        Speak text using gTTS.
        cloud_mode=True → Streamlit audio
        cloud_mode=False → Local playback with pygame
        """
        try:
            tts = gTTS(text=text, lang=lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            if cloud_mode:
                # Streamlit Cloud: play in browser
                import streamlit as st
                st.audio(fp.read(), format="audio/mp3")
            else:
                # Local playback with pygame
                import pygame
                filename = f"temp_{uuid.uuid4().hex}.mp3"
                with open(filename, "wb") as f:
                    f.write(fp.read())

                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                os.remove(filename)

        except Exception as e:
            print(f"[ERROR in TTS]: {e}")

    def ask_gemini(self, detection, model="gemini-2.5-flash"):
        """
        Ask Gemini for navigation instruction.
        Returns short, clear instruction for visually impaired user.
        """
        prompt = f"""
        You are a navigation assistant for a visually impaired person. 
        You only tell the user what is around them and what they should do. 
        Always be short, direct, and clear. 
        Do NOT add extra imagination or stories. 

        Example style:
        - "There is a chair on your left, step carefully."
        - "A person is in front of you, move slowly."
        - "A bed is close, stay cautious."

        Now respond for: {detection}
        """

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            print(f"[ERROR in Gemini]: {e}")
            return detection  # fallback: just repeat detection
