import streamlit as st
import tensorflow as tf
import keras
import pickle
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
st.set_page_config(page_title="Skill Recommender AI", layout="wide")

# --- 1. KONFIGURASI GEMINI ---
# Masukkan API Key kamu di sini atau gunakan environment variable
def get_ai_explanation(job_desc, skills_results):

    skills_names = ", ".join([s for s, score in skills_results])

    prompt = f"""
    Anda adalah AI career assistant profesional.

    Job Description:
    {job_desc}

    Predicted Skills:
    {skills_names}

    Jelaskan:
    1. Kenapa skill tersebut penting
    2. Tips belajar skill tersebut
    3. Prospek karirnya

    Jawab dalam Bahasa Indonesia.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

# --- 2. DEFINISI CUSTOM LAYER ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        attention_weights = tf.nn.softmax(inputs, axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

    def get_config(self):
        return super().get_config()

# --- 3. FUNGSI LOAD DATA & MODEL ---
@st.cache_resource
def load_all_assets():

    model = keras.models.load_model(
        "model/skill_recommendation_model.keras",
        custom_objects={"AttentionLayer": AttentionLayer}
    )

    with open("model/mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    return model, mlb

model, mlb = load_all_assets()

# --- 4. FUNGSI LOGIKA PENDUKUNG ---

def predict_skills(text):
    text_tensor = tf.constant([text])
    pred = model.predict(text_tensor)[0]
    
    # Ambil top 5 skill
    top_indices = np.argsort(pred)[-5:][::-1]
    skills = [mlb.classes_[i] for i in top_indices]
    scores = [float(pred[i]) for i in top_indices]
    
    return list(zip(skills, scores))



# Sidebar untuk informasi tambahan
with st.sidebar:
    st.image("assets/haba_key.jpeg", width=80)
    st.title("Capstone Project")
    st.info("Sistem Rekomendasi Skill & Analisis Tren Lowongan Kerja berbasis Deep Learning.")
    st.write("**Tech Stack:**\n- TensorFlow (Bi-LSTM)\n- Gemini 2.0 Flash\n- Streamlit")

# Main Content
st.title("🎯 AI Skill Recommendation System")
st.write("Gunakan kekuatan Deep Learning dan AI Generatif untuk memahami kebutuhan kompetensi kerja.")

user_input = st.text_area(
    "Masukkan Deskripsi Pekerjaan / Lowongan Kerja:",
    placeholder="Contoh: Dicari Data Scientist yang menguasai Machine Learning dan SQL untuk industri Fintech...",
    height=150
)

results = []

if st.button("Generate Analysis"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks deskripsi terlebih dahulu.")
    else:
        # Proses Prediksi Deep Learning
        results = predict_skills(user_input)
        
        # Layout Kolom untuk Hasil
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("📋 Rekomendasi Skill (Deep Learning)")
            # Tampilkan dalam bentuk progress bar
            for skill, score in results:
                st.write(f"**{skill}**")
                st.progress(score)
                st.caption(f"Confidence Score: {score:.2f}")

        with col2:
            st.subheader("💡 Analisis Kontekstual (Gemini AI)")
            with st.spinner("Gemini sedang menganalisis kaitan skill..."):
                try:
                    explanation = get_ai_explanation(user_input, results)
                    st.markdown(explanation)
                except Exception as e:
                    st.error(f"Gagal terhubung ke Gemini: {e}")

# --- 6. VISUALISASI TREN (ROLE DATA SCIENTIST) ---
st.divider()
st.subheader("📈 Visualisasi Tren Skill (Mockup Data Scientist)")
if results:
    chart_data = {
        "Skill": [s for s, score in results],
        "Demand": [score for s, score in results]
    }

    st.bar_chart(chart_data, x="Skill", y="Demand")