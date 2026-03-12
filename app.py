import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# App Setup
st.set_page_config(page_title="AI Emotion Dashboard", page_icon="🧠", layout="wide")
st.title("Pro AI Emotion Analyzer & Dashboard 🚀")
st.write("Exact emotions-a kandupudikkum & Bulk CSV file-a analyze panni charts podum!")

# 1. Load Exact Emotion AI Model
@st.cache_resource
def load_model():
    # Idhu 7 vidhamana emotions-a kandupudikkum
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

with st.spinner("Pro AI Model load aaguthu... (First time konjam time aagum)"):
    analyzer = load_model()

# Emotion Emojis Dictionary
emotion_emojis = {
    "joy": "😊 Santhosham", "sadness": "😢 Sogam", "anger": "😠 Kobam", 
    "fear": "😨 Bayam", "surprise": "😲 Aachariyam", "disgust": "🤢 Aruvaruppu", "neutral": "😐 Normal"
}

# 2. UI-a Rendu Tabs aaga pirikurom
tab1, tab2 = st.tabs(["✍️ Single Text Analysis", "📁 Bulk CSV Dashboard"])

# --- TAB 1: NORMAL TYPING TEST ---
with tab1:
    st.subheader("Type Panni Test Pannunga")
    user_input = st.text_area("Unga text-a inga type pannunga:")
    
    if st.button("Analyze Emotion"):
        if user_input:
            result = analyzer(user_input)[0]
            label = result['label']
            score = result['score'] * 100
            emoji = emotion_emojis.get(label, "🤔")
            
            st.write("---")
            st.success(f"**Exact Emotion:** {label.capitalize()} - {emoji}")
            st.write(f"**AI Confidence Level:** {score:.2f}%")
            st.progress(int(score))
        else:
            st.warning("Modhalla yethachum type pannunga bro!")

# --- TAB 2: BULK CSV UPLOAD ---
with tab2:
    st.subheader("Upload CSV for Bulk Analysis")
    uploaded_file = st.file_uploader("Oru CSV file upload pannunga", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📂 **Unga Data Preview:**")
        st.dataframe(df.head()) # Modhal 5 rows kaatum
        
        # Entha column-a analyze pannanum nu user kitta kekurom
        text_column = st.selectbox("Entha column-la reviews/text irukku?", df.columns)
        
        if st.button("Run Bulk AI Analysis"):
            with st.spinner("AI ellam rows-yum padikkuthu... Wait pannunga..."):
                emotions = []
                # Oru oru line aah AI kitta anupurom
                for text in df[text_column].astype(str):
                    # Long text iruntha error varama irukka [:512] use pandrom
                    res = analyzer(text[:512])[0] 
                    emotions.append(res['label'].capitalize())
                
                # Pudhusaa Emotion column add pandrom
                df['AI_Emotion'] = emotions
                st.success("Analysis Pakka-va Mudinjithu! 🎉")
                st.dataframe(df) # Result Dataframe
                
                # --- DASHBOARD CHARTS ---
                st.subheader("📊 Emotion Dashboard")
                col1, col2 = st.columns(2)
                
                emotion_counts = df['AI_Emotion'].value_counts().reset_index()
                emotion_counts.columns = ['Emotion', 'Count']
                
                # Pie Chart
                with col1:
                    fig_pie = px.pie(emotion_counts, names='Emotion', values='Count', title="Emotion Percentage", hole=0.3)
                    st.plotly_chart(fig_pie)
                    
                # Bar Chart
                with col2:
                    fig_bar = px.bar(emotion_counts, x='Emotion', y='Count', color='Emotion', title="Total Emotion Count")
                    st.plotly_chart(fig_bar)