import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from newspaper import Article

# --- 1. CONFIGURARE PAGINĂ ---
st.set_page_config(page_title="Detector Deep-Alarmism", page_icon="🔍", layout="wide")

# --- 2. GESTIONARE STARE (SESSION STATE) ---
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def sterge_text():
    st.session_state.input_text = ""

# --- 3. ÎNCĂRCARE MODEL CU CACHE ---
@st.cache_resource
def load_model():
    model_path = "florin-lupsa/NewsAnalyzer" 
    try:
        # Pipeline-ul va descărca automat fișierele de pe Hugging Face la prima rulare online
        return pipeline("text-classification", model=model_path, tokenizer=model_path)
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului de pe Hugging Face: {e}")
        return None

cls_pipeline = load_model()

# --- 4. INTERFAȚA UTILIZATOR ---
st.title("🔍 Analizor de Știri")
st.markdown("Acest instrument analizează atât **titlul**, cât și **conținutul** unui articol pentru a detecta discrepanțe de tip clickbait.")

input_utilizator = st.text_input(
    "Introdu un Titlu sau un Link către un articol:", 
    key="input_text"
)

# Coloane pentru butoanele principale
col_btn1, col_btn2 = st.columns([1, 1])

analizeaza = False
with col_btn1:
    if st.button("🚀 Analizează", use_container_width=True, type="primary"):
        analizeaza = True

with col_btn2:
    if st.session_state.input_text != "":
        st.button("🧹 Reset / Golește", on_click=sterge_text, use_container_width=True)

# --- 5. LOGICA DE PROCESARE ---
if analizeaza and input_utilizator:
    titlu_final = ""
    text_articol = ""
    
    # Verificăm dacă este link sau text simplu
    if input_utilizator.startswith("http"):
        try:
            with st.spinner('Se descarcă și se analizează articolul complet...'):
                articol = Article(input_utilizator)
                articol.download()
                articol.parse()
                
                titlu_final = articol.title
                # Luăm primele 1200 de caractere pentru a nu bloca modelul BERT
                text_articol = articol.text[:1200]
                
                st.info(f"**Titlu detectat:** {titlu_final}")
        except Exception as e:
            st.error(f"Eroare la procesarea link-ului: {e}")
    else:
        titlu_final = input_utilizator
        # În cazul textului simplu, nu avem conținut separat

# --- 6. ANALIZA AI ---
    if titlu_final:
        # Analiză Titlu
        rez_titlu = cls_pipeline(titlu_final)[0]
        scor_titlu = rez_titlu['score'] * 100
        
        # Dacă avem și text de articol, facem Deep Analysis
        if text_articol:
            rez_text = cls_pipeline(text_articol)[0]
            scor_text = rez_text['score'] * 100
            
            # Afișăm metricile comparativ
            st.divider()
            m1, m2 = st.columns(2)
            with m1:
                # Modificat: Afișăm încrederea în funcție de ce etichetă a ales modelul
                tip_t = "Alarmism" if rez_titlu['label'] == "LABEL_1" else "Informativ"
                st.metric(label=f"Certitudine {tip_t} (Titlu)", value=f"{scor_titlu:.1f}%")
            with m2:
                tip_x = "Alarmism" if rez_text['label'] == "LABEL_1" else "Informativ"
                st.metric(label=f"Certitudine {tip_x} (Conținut)", value=f"{scor_text:.1f}%")
            
            # Verificăm discrepanța (Clickbait)
            if rez_titlu['label'] == "LABEL_1" and rez_text['label'] == "LABEL_0":
                st.warning("⚠️ **DETECȚIE CLICKBAIT:** Titlul este disproporționat de alarmist față de textul articolului!")
            
            # Scor final mediu
            scor_final = (scor_titlu + scor_text) / 2
            # label_final = "LABEL_1" if scor_final > 50 else "LABEL_0"
            # NOU: Verdictul se bazează pe etichete. Dacă ambele sunt informative, finalul e informativ.
            if rez_titlu['label'] == "LABEL_1" or rez_text['label'] == "LABEL_1":
                label_final = "LABEL_1"
            else:
                label_final = "LABEL_0"
        else:
            # Dacă e doar text simplu
            scor_final = scor_titlu
            label_final = rez_titlu['label']

        # --- 7. AFIȘARE VERDICT FINAL ---
        st.divider()
        if label_final == "LABEL_1":
            st.error("**Verdict Final: ALARMIST 🚩**")
            st.markdown(f"Nivel de încredere al modelului: **{scor_final:.2f}%**")
            st.progress(scor_final / 100)
            st.caption("Notă: Acest procent reprezintă gradul de certitudine al AI-ului privind stilul senzaționalist, nu veridicitatea faptelor.")
        else:
            st.success("**Verdict Final: INFORMATIV ✅**")
            st.markdown(f"Nivel de încredere al modelului: **{scor_final:.2f}%**")
            st.progress(scor_final / 100)
    else:
        st.warning("Te rugăm să introduci un conținut valid.")
# --- 8. SIDEBAR ---
st.sidebar.title("Despre Proiect")
st.sidebar.info("Acest detector folosește un model **BERT Romanian** antrenat să facă distincția între jurnalismul factual și cel senzaționalist.")
st.sidebar.markdown("---")
st.sidebar.write("📌 **Analiza:** Când introduci un link, sistemul compară titlul cu primele paragrafe pentru a identifica manipularea prin clickbait.")
