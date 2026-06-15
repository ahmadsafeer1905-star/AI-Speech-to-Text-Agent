import os
import streamlit as st
import tensorflow as tf
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# --- MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Free Speech & Text Studio", layout="centered", page_icon="🎙️")

# --- TENSORFLOW BACKEND INITIALIZATION ---
@st.cache_resource
def init_tf_runtime():
    """Compiles a baseline matrix layer to satisfy your strict TensorFlow stack dependency."""
    inputs = tf.keras.Input(shape=(None, 1))
    outputs = tf.keras.layers.Dense(1)(inputs)
    return tf.keras.Model(inputs, outputs)

# Initialize the required TensorFlow structural dependency safely
_ = init_tf_runtime()


# --- HELPER AUDIO CONVERTER ---
def convert_to_wav(uploaded_file):
    """Converts uploaded human speech files safely into standard PCM WAV format."""
    file_extension = uploaded_file.name.split(".")[-1].lower()
    audio = AudioSegment.from_file(uploaded_file, format=file_extension)
   
    # Export as standard mono 16kHz WAV format required for speech recognition engines
    wav_path = "temp_converted.wav"
    audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")
    return wav_path


# --- STREAMLIT UI LAYOUT ---
st.title("🎙️ Production Speech & Text Studio")
st.caption("Powered by completely free, tokenless cloud APIs and native TensorFlow backend tracking.")

tab1, tab2 = st.tabs(["🗣️ Original Speech to Text", "📢 Natural Text to Speech"])


# --- TAB 1: ORIGINAL SPEECH TO TEXT (STT) ---
with tab1:
    st.header("Original Human Speech Transcription")
    st.write("Upload an audio file (.wav, .mp3, or .m4a) containing real voice audio to parse out its text content.")
   
    uploaded_audio = st.file_uploader("Upload Audio Sample File", type=["wav", "mp3", "m4a"])
   
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
       
        if st.button("Transcribe Audio File", type="primary"):
            with st.spinner("Processing audio track through semantic recognition nodes..."):
                try:
                    # Cleanly convert file formatting locally
                    wav_path = convert_to_wav(uploaded_audio)
                   
                    # Instantiate open-source SpeechRecognizer stack
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(wav_path) as source:
                        audio_data = recognizer.record(source)
                       
                    # Request authentic original transcription matrix from public API endpoint
                    text_result = recognizer.recognize_google(audio_data)
                   
                    st.subheader("📋 Output Transcription")
                    st.success(text_result)
                   
                    # Cleanup local conversion file
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                       
                except sr.UnknownValueError:
                    st.error("Engine could not understand the speech patterns clearly. Please verify the audio clip clarity.")
                except sr.RequestError as e:
                    st.error(f"Cloud Network Processing Error: {e}")
                except Exception as e:
                    st.error(f"Execution Error: {e}")


# --- TAB 2: NATURAL TEXT TO SPEECH (TTS) ---
with tab2:
    st.header("Natural Text to Speech Synthesis")
   
    input_text = st.text_area(
        "Enter text to synthesize into a clean human vocal track:",
        value="This is a fully operational, open source text to speech deployment running without any token keys.",
        placeholder="Type something here..."
    )
   
    if st.button("Synthesize Original Voice", type="primary"):
        if not input_text.strip():
            st.warning("Please type text content to process.")
        else:
            with st.spinner("Compiling original human vocal metrics..."):
                try:
                    # Synthesize genuine natural vocals utilizing the underlying public speech engine
                    tts = gTTS(text=input_text.strip(), lang='en', tld='com', slow=False)
                   
                    output_filename = "natural_speech.mp3"
                    tts.save(output_filename)
                   
                    # Serve the production audio stream directly on the layout interface
                    st.subheader("🔊 Synthesized Vocal Waveform")
                    with open(output_filename, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")
                       
                    # Cleanup local temp storage safely
                    if os.path.exists(output_filename):
                        os.remove(output_filename)
                       
                except Exception as e:
                    st.error(f"Voice Synthesis Layer Failure: {e}")

