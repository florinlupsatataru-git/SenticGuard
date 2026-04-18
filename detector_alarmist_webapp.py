import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from newspaper import Article

# --- 1. CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="Detector Deep-Alarmism", page_icon="🔍", layout="wide")

# --- 2. GESTIONARE STARE ---
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def sterge_text():
    st.session_state.input_text = ""

# --- 3. ÎNCĂRCARE MODEL CU CACHE ---
@st.cache_resource
def load_model():
    # MODIFICAT: Calea către Hugging Face
    model_path = "florin-lupsa/NewsAnalyzer" 
    try:
        # Pipeline-ul descarcă automat de pe HF dacă nu găsește local
        return pipeline("text-classification", model=model_path, tokenizer=model_path)
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului: {e}")
        return None

cls_pipeline = load_model()

# --- 4. INTERFAȚA UTILIZATOR ---
st.title("🔍 Analizor de Știri - Deep Analysis")
st.markdown("Analizează **titlul** și **conținutul** pentru a detecta alarmismul și discrepanțele de tip clickbait.")

input_text = st.text_area("Introdu link-ul știrii sau textul complet:", 
                          value=st.session_state.input_text, 
                          height=150, key="input_area")

col1, col2 = st.columns([1, 5])
with col1:
    buton_analiza = st.button("🚀 Analizează")
with col2:
    st.button("🗑️ Șterge tot", on_click=sterge_text)

# --- 5. LOGICA DE ANALIZĂ ---
if buton_analiza:
    if input_text:
        text_to_check = input_text
        content_extracted = False
        
        # Extracție conținut dacă e URL
        if input_text.startswith(("http://", "https://")):
            with st.spinner("Se extrage conținutul articolului..."):
                try:
                    article = Article(input_text)
                    article.download()
                    article.parse()
                    title = article.title
                    text = article.text
                    content_extracted = True
                    st.info(f"**Titlu Detectat:** {title}")
                except Exception as e:
                    st.error(f"Nu s-a putut accesa link-ul. Eroare: {e}")
                    text_to_check = input_text

        # --- 6. ANALIZĂ AI ---
        with st.spinner("AI-ul analizează stilul limbajului..."):
            if content_extracted:
                # Analizăm titlul și textul separat
                rez_titlu = cls_pipeline(title)[0]
                rez_text = cls_pipeline(text)[0]
                
                scor_titlu = rez_titlu['score'] * 100
                label_titlu = rez_titlu['label']
                
                scor_text = rez_text['score'] * 100
                label_text = rez_text['label']
            else:
                # Analizăm doar input-ul ca titlu/text unitar
                rez_titlu = cls_pipeline(text_to_check)[0]
                scor_titlu = rez_titlu['score'] * 100
                label_titlu = rez_titlu['label']
                label_text = None

        # --- 7. AFIȘARE REZULTATE DETALIATE ---
        st.divider()
        
        # Secțiune Titlu
        if label_titlu == "LABEL_1":
            st.error(f"🚩 **TITLU ALARMIST** (Încredere: {scor_titlu:.2f}%)")
        else:
            st.success(f"✅ **TITLU INFORMATIV** (Încredere: {scor_titlu:.2f}%)")

        if content_extracted:
            # Secțiune Conținut
            if label_text == "LABEL_1":
                st.error(f"🚩 **CONȚINUT ALARMIST** (Încredere: {scor_text:.2f}%)")
            else:
                st.success(f"✅ **CONȚINUT INFORMATIV** (Încredere: {scor_text:.2f}%)")
            
            # Detecție Clickbait (Dacă titlul e alarmist dar textul e informativ)
            if label_titlu == "LABEL_1" and label_text == "LABEL_0":
                st.warning("⚠️ **ALERTA CLICKBAIT:** Titlul este alarmist, deși conținutul este unul neutru/informativ!")
            
            # Calcul Verdict Final
            # Dacă oricare dintre ele e LABEL_1, considerăm un risc de alarmism
            if label_titlu == "LABEL_1" or label_text == "LABEL_1":
                label_final = "LABEL_1"
                scor_final = max(scor_titlu, scor_text)
            else:
                label_final = "LABEL_0"
                scor_final = (scor_titlu + scor_text) / 2
        else:
            label_final = label_titlu
            scor_final = scor_titlu

        # --- 8. AFIȘARE VERDICT FINAL ---
        st.divider()
        if label_final == "LABEL_1":
            st.error("### **Verdict Final: ALARMIST 🚩**")
            st.write(f"Gradul de certitudine al AI-ului: **{scor_final:.2f}%**")
            st.progress(scor_final / 100)
        else:
            st.success("### **Verdict Final: INFORMATIV ✅**")
            st.write(f"Gradul de certitudine al AI-ului: **{scor_final:.2f}%**")
            st.progress(scor_final / 100)
    else:
        st.warning("Te rugăm să introduci un link sau un text pentru analiză.")

# --- 9. SIDEBAR ---
st.sidebar.title("Despre Proiect")
st.sidebar.info("Modelul folosește un transformer BERT antrenat special pe știri din România pentru a distinge între jurnalism informativ și clickbait/alarmism.")
