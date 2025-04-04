import streamlit as st
import numpy as np
import torch
import os

from utils import *
from model.model import CVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_SAMPLES_DIR = "audios_samples"

st.set_page_config(
    page_title="Audio Reconstruction",
    page_icon="./images/aivn_favicon.png",
)

st.image("./images/aivn_logo.png", width=300)

st.title("New Genres Audio Reconstruction")

@st.cache_data
def load_models():
    st.spinner("Loading model...")
    # Cache the model to avoid reloading
    model = CVAE(64, 128, 256, 130, len(uni_genres_list)).to(device)
    model.load_state_dict(torch.load('model/model_256.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


def gen_audio(model, audio_source, genres_list, fixed_length_seconds=3):
    with st.spinner("Processing audio..."):
        audio_data, sr = load_and_resample_audio(audio_source)
        n_frames = len(audio_data)
        segment_length_frame = int(fixed_length_seconds * sr)
        n_segments = n_frames // segment_length_frame

        split_audio_text_placeholder = st.empty()
        split_audio_text_placeholder.text("Splitting audio into segments... ✂")
        progress_bar_placeholder = st.empty()
        progress_bar = progress_bar_placeholder.progress(0)

        audios = []
        for i in range(n_segments):
            start = i * segment_length_frame
            end = (i + 1) * segment_length_frame
            segment = audio_data[start:end]
            mel_spec = audio_to_melspec(segment, sr, to_db=True)
            mel_spec_norm = normalize_melspec(mel_spec)
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
            mel_spec_norm = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audios.append((mel_spec_norm, mel_spec))
            progress_bar.progress(int((i + 1) / n_segments * 100))

        progress_bar_placeholder.empty()
        split_audio_text_placeholder.empty()

        audios_input = torch.cat([audio[0] for audio in audios], dim=0)
        audios_input = audios_input.to(device)

        # Encode genres
        genres_encoded = onehot_encode(tokenize(genres_list), len(uni_genres_list))
        genres_input = torch.tensor(genres_encoded, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        genres_input = genres_input.repeat(audios_input.shape[0], 1, 1)
        genres_input = genres_input.to(device)


        with st.spinner("Model is generating... 🍳"):
            recons, _, _ = model(audios_input, genres_input)

        recon_audio_text_placeholder = st.empty()
        recon_audio_text_placeholder.text("Reconstructing audio... 🎵")
        progress_bar_placeholder = st.empty()
        progress_bar = progress_bar_placeholder.progress(0)
        recon_audios = []

        for i in range(len(recons)):
            spec_denorm = denormalize_melspec(recons[i].detach().cpu().numpy().squeeze(), audios[i][1])
            audio_reconstructed = melspec_to_audio(spec_denorm)
            recon_audios.append(audio_reconstructed)
            progress_bar.progress(int((i + 1) / len(recons) * 100))

        recon_audios = np.concatenate(recon_audios)
        progress_bar_placeholder.empty()
        recon_audio_text_placeholder.empty()

        return recon_audios


def run():
    model = load_models()
    uploaded_audio = st.file_uploader("Upload an audio file (only the first 15 seconds will be processed)", type=['wav', 'mp3'])

    select_audio = st.selectbox(
        "Or select a sample audio below:",
        options=[""] + [f"{file} - from GTZAN Dataset" for file in os.listdir(AUDIO_SAMPLES_DIR) if file.endswith(('.wav', '.mp3'))],
        index=0,
        format_func=lambda x: "No sample audio selected" if x == "" else x
    )

    if uploaded_audio is not None or select_audio != "":
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')
        else:
            uploaded_audio = os.path.join(AUDIO_SAMPLES_DIR, select_audio.replace(" - from GTZAN Dataset", ""))
            st.audio(uploaded_audio, format='audio/wav')

        genres_list = st.multiselect('Select genres', uni_genres_list)

        if st.button('Process Audio'):
            result = gen_audio(model, uploaded_audio, genres_list)
            st.write('Reconstructed Result:')
            st.audio(result, format='audio/wav', sample_rate=22050)

run()

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>

    <div class="footer">
        <div>
            <a href="https://ieeexplore.ieee.org/document/1021072">*GTZAN Dataset</a>
        </div>
        <div>
            2024 AI VIETNAM | Created by <a href="https://github.com/Koii2k3/Music-Reconstruction" target="_blank">Koii2k3</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
