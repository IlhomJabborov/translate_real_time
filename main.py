import streamlit as st
import sounddevice as sd
import torchaudio
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torch

# Load the model and processor
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Model sampling rate (16 kHz)
SAMPLING_RATE = 16_000
CHUNK_DURATION = 2  # seconds
CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DURATION)  # Number of samples per chunk

# Streamlit UI
st.title("Real-time Audio Translation")
st.write("Please select the target language and press the button to start translating.")

target_language = st.selectbox("Select Target Language:", ["uzn", "eng"])  # Add more languages as needed

if st.button("Start Translation"):
    st.write("Real-time translation started. Speak into your microphone...")

    # Function to process audio and generate translation
    def process_audio_chunk(audio_chunk, lang_tg):
        audio_inputs = processor(audios=audio_chunk, return_tensors="pt")
        audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=lang_tg)[0].cpu().numpy().squeeze()
        return audio_array_from_audio

    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        # Resample audio to 16 kHz if needed
        audio_resampled = torchaudio.functional.resample(torch.tensor(indata.T), orig_freq=SAMPLING_RATE, new_freq=SAMPLING_RATE)
        # Process audio chunk
        translated_audio = process_audio_chunk(audio_resampled, target_language)
        # Play back the translated audio
        sd.play(translated_audio, samplerate=model.config.sampling_rate)

    # Start the audio stream
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLING_RATE, blocksize=CHUNK_SIZE):
        try:
            while True:
                pass
        except KeyboardInterrupt:
            st.write("Real-time translation stopped.")