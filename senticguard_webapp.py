import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from newspaper import Article

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="SenticGuard AI: Ethical Integrity Check", page_icon="🔍", layout="wide")

# --- 2. CONFIGURAȚIE CATEGORII (Mapping & UI) ---
CATEGORIES = {
    "OBIECTIV": {"color": "#28a745", "icon": "✅", "desc": "Știre neutră, bazată pe fapte și date verificabile."},
    "ALARMIST": {"color": "#dc3545", "icon": "🚩", "desc": "Conținut care încearcă să inducă teamă sau panică nejustificată."},
    "SENZATIONAL": {"color": "#fd7e14", "icon": "💥", "desc": "Titlu de tip clickbait conceput pentru a atrage click-uri prin curiozitate."},
    "CONFLICTUAL": {"color": "#6f42c1", "icon": "⚔️", "desc": "Știre care alimentează scandalul, revolta sau polarizarea socială."},
    "INFORMATIV": {"color": "#007bff", "icon": "ℹ️", "desc": "Conținut utilitar, ghiduri practice, rețete sau sfaturi."},
    "OPINIE": {"color": "#6c757d", "icon": "✍️", "desc": "Perspectivă subiectivă, editorial sau analiză personală."}
}

# --- 3. SESSION STATE ---
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

def sterge_text():
    st.session_state.input_text = ""

# --- 4. MODEL LOAD WITH CACHE ---
@st.cache_resource
def load_model():
    model_path = "florin-lupsa/NewsAnalyzer" 
    try:
        return pipeline("text-classification", model=model_path, tokenizer=model_path)
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului: {e}")
        return None

cls_pipeline = load_model()

# --- 5. LOGIC FOR BROWSER EXTENSION ---
query_params = st.query_params
if "predict_text" in query_params:
    text_to_analyze = query_params["predict_text"]
    if cls_pipeline:
        prediction = cls_pipeline(text_to_analyze[:512])[0]
        # Trimitem JSON-ul înapoi către extensie
        st.json({
            "label": prediction['label'],
            "score": float(prediction['score']),
            "color": CATEGORIES.get(prediction['label'], {"color": "#333"})["color"]
        })
        st.stop()

# --- UI PRINCIPALĂ ---
st.title("🛡️ SenticGuard AI")
st.markdown("### Detector de Integritate și Stil Jurnalistic v2.0")
st.write("Analizează titluri sau articole pentru a identifica intenția și tonul comunicării.")

# --- 6. INPUT SECTION ---
input_mode = st.radio("Alege metoda de analiză:", ["Titlu / Text manual", "Link Articol (URL)"])

if input_mode == "Link Articol (URL)":
    url = st.text_input("Introdu URL-ul știrii:")
    if url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            titlu_analiza = article.title
            text_analiza = article.text
            st.info(f"**Titlu detectat:** {titlu_analiza}")
        except Exception as e:
            st.error(f"Nu s-a putut procesa URL-ul: {e}")
            titlu_analiza, text_analiza = "", ""
else:
    titlu_analiza = st.text_area("Introdu titlul sau textul aici:", value=st.session_state.input_text, height=150)
    text_analiza = ""

# --- 7. ANALIZĂ ---
if st.button("Analizează Conținutul"):
    if titlu_analiza and cls_pipeline:
        with st.spinner('SenticGuard analizează textul...'):
            # Analiză Titlu
            rez_titlu = cls_pipeline(titlu_analiza[:512])[0]
            label_final = rez_titlu['label']
            scor_final = rez_titlu['score'] * 100

            # Dacă avem și text lung din URL, îl analizăm și pe acela
            if text_analiza:
                rez_text = cls_pipeline(text_analiza[:512])[0]
                # Prioritizăm titlul dacă acesta este marcat ca alarmist/senzațional, 
                # altfel facem o pondere (titlul contează 70%)
                if rez_titlu['label'] in ["ALARMIST", "SENZATIONAL", "CONFLICTUAL"]:
                    label_final = rez_titlu['label']
                    scor_final = rez_titlu['score'] * 100
                else:
                    label_final = rez_titlu['label']
                    scor_final = (rez_titlu['score'] * 0.7 + rez_text['score'] * 0.3) * 100

        # --- 8. AFIȘARE VERDICT ---
        st.divider()
        
        cat_data = CATEGORIES.get(label_final, {"color": "#333", "icon": "❓", "desc": "Categorie necunoscută"})

        st.markdown(f"""
            <div style="background-color: {cat_data['color']}; padding: 30px; border-radius: 15px; color: white; text-align: center;">
                <h1 style="margin: 0; font-size: 32px;">{cat_data['icon']} Verdict: {label_final}</h1>
                <p style="margin: 10px 0 0 0; font-size: 20px; opacity: 0.9;">{cat_data['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### Scichetă Detectată: **{label_final}**")
        with col2:
            st.write(f"### Încredere Model: **{scor_final:.2f}%**")
            st.progress(scor_final / 100)

        st.caption("🔍 **Notă:** SenticGuard analizează amprenta psihologică și stilul lingvistic. Acest verdict nu reprezintă o confirmare a adevărului absolut, ci a modului în care informația este prezentată.")

    else:
        st.warning("Te rugăm să introduci un conținut valid pentru analiză.")

# --- 9. SIDEBAR ---
st.sidebar.title("Legenda Categoriilor")
for cat, info in CATEGORIES.items():
    st.sidebar.markdown(f"**{info['icon']} {cat}**: {info['desc']}")

st.sidebar.divider()
st.sidebar.button("Șterge tot", on_click=sterge_text)
st.sidebar.info("SenticGuard v2.0 powered by Multilingual BERT")
