import streamlit as st
import random
import pandas as pd
import psycopg2
import requests
import json
from transformers import pipeline
from newspaper import Article, Config
from streamlit_gsheets import GSheetsConnection
from senticguard_translations import TRANSLATIONS

# --- 1. CONFIGURATION & CONSTANTS ---
WEIGHT_CONTENT = 0.7 
WEIGHT_TITLE = 0.3

st.set_page_config(
    page_title="SenticGuard AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1.1 GEMINI NATIVE HTTP API RESOLVER (FĂRĂ ARANJAMENTE SUB PREȘ) ---
def generate_dynamic_explanation(title, content, verdict_label, lang):
    """
    Generates a dynamic 2-sentence explanation using a rock-solid HTTP request.
    NO MORE SILENT CATCHES. If it fails, it prints the real raw network logs.
    """
    # Safe secret fetching
    if "gemini_api" in st.secrets and "api_key" in st.secrets["gemini_api"]:
        api_key = st.secrets["gemini_api"]["api_key"]
    elif "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        api_key = st.secrets["gemini"]["api_key"]
    else:
        api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("❌ Eroare critică: Cheia API Gemini lipsește complet din secrets.toml!")
        return None

    context_text = f"Titlu: {title}"
    if content and content.strip():
        context_text += f"\nContinut: {content[:1000]}"

    if lang == "RO":
        prompt = (
            f"Explică pe un ton sociologic și echilibrat, în maximum două propoziții scurte, "
            f"de ce textul următor a fost clasificat drept '{verdict_label}'.\n\n"
            f"{context_text}\n\n"
            f"Nu folosi introduceri precum 'Acest articol...', mergi direct la subiectul analizei discursului."
        )
    else:
        prompt = (
            f"Explain in a professional sociological tone, using a maximum of two short sentences, "
            f"why the following text was classified as '{verdict_label}'.\n\n"
            f"{context_text}\n\n"
            f"Do not use conversational intros like 'This article...', go straight to the discourse analysis."
        )

    # Clean, verified production URL
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    # We pass the key as a clean query parameter dictionary, preventing parsing breaks
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    # No try-except here. Let it scream if it fails!
    response = requests.post(url, headers=headers, json=payload, params=params, timeout=10)
    
    if response.status_code == 200:
        res_json = response.json()
        return res_json['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        # This will render EXACTLY in the middle of the screen if Google rejects us
        st.error(f"💥 Google API Server Refusal! Status: {response.status_code} - Response Text: {response.text}")
        return None

# --- 2. DATABASE LOGGING (PostgreSQL) ---
def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        port=st.secrets["postgres"]["port"]
    )

def log_security_event(event_type="VISIT_8501", severity=1, description="Direct access to Streamlit interface"):
    try:
        headers = st.context.headers
        ip_addr = "Unknown"
        if "X-Forwarded-For" in headers:
            ip_addr = headers["X-Forwarded-For"].split(",")[0].strip()
        elif "X-Real-IP" in headers:
            ip_addr = headers["X-Real-IP"]
            
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO security_logs (ip_address, event_type, severity, description) VALUES (%s, %s, %s, %s)",
            (ip_addr, event_type, severity, description)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass

def log_analysis_to_gsheets(url_or_text, source_domain, verdict, confidence):
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        df_existing = conn.read(worksheet="Logs", ttl=0)
        new_row = pd.DataFrame([{
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_data": url_or_text[:255],
            "source": source_domain,
            "verdict": verdict,
            "confidence": float(confidence)
        }])
        df_updated = pd.concat([df_existing, new_row], ignore_index=True)
        conn.update(worksheet="Logs", data=df_updated)
    except Exception:
        pass

# --- 3. LANGUAGE AND SESSION INITIALIZATION ---
if "lang" not in st.session_state:
    st.session_state["lang"] = "RO"

if "logged_visit" not in st.session_state:
    log_security_event("VISIT", 1, "Redirect to AI Interface")
    st.session_state["logged_visit"] = True

# --- 4. LANGUAGE SELECTOR ---
col_space, col_lang = st.columns([0.85, 0.15])
with col_lang:
    lang_choice = st.selectbox("🌐 Language", ["Română", "English"], index=0 if st.session_state["lang"] == "RO" else 1)
    st.session_state["lang"] = "RO" if lang_choice == "Română" else "EN"

T = TRANSLATIONS[st.session_state["lang"]]

# --- 5. MODEL CACHING ---
@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="./model_temp", tokenizer="./model_temp")

classifier = load_classifier()

# --- 6. CATEGORIES STYLING AND DICTIONARY ---
VERDICT_INFO = {
    0: {"label": "OBIECTIV", "class": "card-objective"},
    1: {"label": "ALARMIST", "class": "card-alarmist"},
    2: {"label": "SENZATIONAL", "class": "card-senzational"},
    3: {"label": "CONFLICTUAL", "class": "card-conflictual"},
    4: {"label": "INFORMATIV", "class": "card-informativ"},
    5: {"label": "OPINIE", "class": "card-opinie"}
}

# --- 7. MAIN UI DESIGN ---
st.markdown(f'<h1 class="main-title">{T["main_title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{T["sub_title"]}</p>', unsafe_allow_html=True)

st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 2.8rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; font-size: 1.2rem; color: #64748b; margin-bottom: 2.5rem; }
    .card-result { border-radius: 12px; padding: 1.8rem; margin-top: 1.5rem; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05); }
    .card-objective { background-color: #f0fdf4; border-left: 6px solid #16a34a; }
    .card-alarmist { background-color: #fef2f2; border-left: 6px solid #dc2626; }
    .card-senzational { background-color: #fffbeb; border-left: 6px solid #d97706; }
    .card-conflictual { background-color: #faf5ff; border-left: 6px solid #9333ea; }
    .card-informativ { background-color: #eff6ff; border-left: 6px solid #2563eb; }
    .card-opinie { background-color: #f8fafc; border-left: 6px solid #475569; }
    .badge-verdict { display: inline-block; padding: 0.25rem 0.75rem; font-size: 0.85rem; font-weight: 700; border-radius: 20px; text-transform: uppercase; margin-bottom: 0.75rem; }
    .badge-objective { background-color: #dcfce7; color: #16a34a; }
    .badge-alarmist { background-color: #fee2e2; color: #dc2626; }
    .badge-senzational { background-color: #fef3c7; color: #d97706; }
    .badge-conflictual { background-color: #f3e8ff; color: #9333ea; }
    .badge-informativ { background-color: #dbeafe; color: #2563eb; }
    .badge-opinie { background-color: #f1f5f9; color: #475569; }
    .article-title { font-size: 1.4rem; font-weight: 700; color: #0f172a; margin-top: 0; margin-bottom: 0.75rem; line-height: 1.4; }
    </style>
""", unsafe_allow_html=True)

# --- 8. INPUT PROCESSING (URL VS TEXT) ---
input_mode = st.radio(T["lang_select"], [T["tab_link"], T["tab_manual"]], horizontal=True)

input_data = ""
if input_mode == T["tab_link"]:
    input_data = st.text_input(T["url_label"], placeholder="https://www.digi24.ro/...")
else:
    input_data = st.text_area(T["manual_label"], placeholder="Introduceți textul aici...")

# --- 9. ANALYSIS LOGIC ---
if st.button(T["analyze_btn"], type="primary"):
    if not input_data.strip():
        st.warning(T["warn_no_input"])
    else:
        titlu_analiza = ""
        text_analiza = ""
        source_domain = "Direct Text Input"
        
        if input_mode == T["tab_link"]:
            with st.spinner(T["success_load"]):
                try:
                    config = Config()
                    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...'
                    article = Article(input_data, config=config)
                    article.download()
                    article.parse()
                    titlu_analiza = article.title
                    text_analiza = article.text
                    source_domain = input_data.split("/")[2].replace("www.", "")
                except Exception as e:
                    st.error(f"{T['error_load']}: {e}")
                    st.stop()
        else:
            titlu_analiza = input_data
            text_analiza = ""

        with st.spinner(T["analyze_btn"] + "..."):
            res_titlu = classifier(titlu_analiza[:512])[0]
            label_title_str = res_titlu['label']
            score_title = res_titlu['score']
            
            res_content = None
            if text_analiza.strip():
                res_content = classifier(text_analiza[:512])[0]
                label_content_str = res_content['label']
                score_content = res_content['score']

            label_map = {"OBIECTIV":0, "ALARMIST":1, "SENZATIONAL":2, "CONFLICTUAL":3, "INFORMATIV":4, "OPINIE":5}
            id_title = label_map[label_title_str]
            
            if res_content:
                id_content = label_map[label_content_str]
                if id_title == id_content:
                    final_id = id_title
                    final_score = (score_title * WEIGHT_TITLE) + (score_content * WEIGHT_CONTENT)
                else:
                    if score_content >= 0.65:
                        final_id = id_content
                        final_score = score_content
                    else:
                        final_id = id_title
                        final_score = score_title
            else:
                final_id = id_title
                final_score = score_title

            localized_verdict = T.get(f"label_{final_id}", VERDICT_INFO[final_id]["label"])

            trans_templates = T.get("template") or T.get("templates") or {}
            is_match = not res_content or id_title == label_map[res_content['label']]
            branch_key = "match" if is_match else "mismatch"
            
            if trans_templates and branch_key in trans_templates:
                static_list = trans_templates[branch_key]
            else:
                if st.session_state["lang"] == "RO":
                    static_list = ["Analiza sistemului indică un stil preponderent {label_v}."]
                else:
                    static_list = ["System analysis indicates a predominantly {label_v} style."]

            fallback_explanation = random.choice(static_list).format(
                label_s=label_title_str, 
                label_v=localized_verdict
            )

            verdict_final = {
                "id": final_id,
                "label": localized_verdict,
                "class": VERDICT_INFO[final_id]["class"],
                "score": final_score,
                "explanation": fallback_explanation
            }

            # --- DYNAMIC GEMINI EXPLANATION GENERATION ---
            dynamic_exp = generate_dynamic_explanation(
                titlu_analiza, 
                text_analiza, 
                VERDICT_INFO[final_id]["label"], 
                st.session_state["lang"]
            )
            
            if dynamic_exp:
                verdict_final["explanation"] = dynamic_exp

        log_security_event("VERIFY", 1, f"Analyzed item from {source_domain}. Result: {VERDICT_INFO[final_id]['label']}")
        log_analysis_to_gsheets(titlu_analiza, source_domain, VERDICT_INFO[final_id]['label'], verdict_final['score'])

        # --- 10. RESULTS RENDERING (HTML TEMPLATE) ---
        verdict_prefix_text = T.get('verdict_prefix', 'VERDICT:')
        st.markdown(f"""
            <div class="card-result {verdict_final['class']}">
                <div class="badge-verdict badge-{verdict_final['class'].split('-')[1]}">
                    {verdict_prefix_text} {verdict_final['label']}
                </div>
                <h3 class=\"article-title\">{titlu_analiza}</h3>
                <p style="font-size: 1.1rem; color: #334155; font-weight: 500; line-height: 1.5; margin-bottom: 0;">
                    {verdict_final['explanation']}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # TECHNICAL DETAILS EXPANDER
        with st.expander(T['deep_title']):
            col_r1, col_r2 = st.columns(2)
            with col_r1: 
                st.metric(T["tech_manual_label"], res_titlu['label'])
                st.progress(res_titlu['score'], text=f"{T['confidence']} {res_titlu['score']:.2%}")
            
            with col_r2: 
                if res_content:
                    st.metric(T["tech_content_label"], res_content['label'])
                    st.progress(res_content['score'], text=f"{T['confidence']} {res_content['score']:.2%}")
                else:
                    st.info(T["tech_no_content"])
            
            st.divider()
            st.write(f"**{T['tech_final_label']}** {verdict_final['score']:.2%}")

# --- 11. SIDEBAR LEGEND ---
with st.sidebar:
    st.markdown(f"### 📋 {T['sidebar_title']}")
    st.markdown(T["system_desc"])
