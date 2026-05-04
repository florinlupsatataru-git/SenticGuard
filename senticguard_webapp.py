import streamlit as st
from transformers import pipeline
from newspaper import Article, Config

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="SenticGuard AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded" # Am schimbat aici să fie vizibil
)

# --- 2. MINIMAL DESIGN (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .verdict-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        color: white;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CATEGORIES CONFIG ---
CATEGORIES = {
    "OBIECTIV": {"color": "#10b981", "desc": "Informație neutră, bazată pe fapte verificabile."},
    "ALARMIST": {"color": "#ef4444", "desc": "Titlu care induce panică sau teamă exagerată."},
    "CLICKBAIT": {"color": "#f59e0b", "desc": "Creat special pentru a forța utilizatorul să dea click."},
    "CONFLICTUAL": {"color": "#8b5cf6", "desc": "Subliniază dispute, certuri sau tensiuni sociale."},
    "INFORMATIV": {"color": "#3b82f6", "desc": "Conținut util, de tip ghid sau explicații practice."},
    "OPINIE": {"color": "#64748b", "desc": "Punct de vedere subiectiv sau analiză personală."}
}

# --- 4. MODEL LOAD ---
@st.cache_resource
def load_model():
    model_path = "florin-lupsa/NewsAnalyzer" 
    try:
        return pipeline("text-classification", model=model_path, tokenizer=model_path)
    except Exception as e:
        st.error(f"Eroare la încărcarea modelului: {e}")
        return None

cls_pipeline = load_model()

def analyze_text(text):
    if not text or not cls_pipeline:
        return None
    # Curățăm textul de spații inutile înainte de analiză
    prediction = cls_pipeline(text.strip()[:512])[0]
    return {
        "label": prediction['label'],
        "score": float(prediction['score']),
        "config": CATEGORIES.get(prediction['label'], {"color": "#64748b", "desc": ""})
    }

# --- 5. INTERFACE ---
st.title("SenticGuard AI")
st.markdown("#### Media Integrity & Deep Analysis")

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    input_mode = st.tabs(["Link Articol", "Text Manual"])
    
    titlu_analiza = ""
    text_analiza = ""

    with input_mode[0]:
        url = st.text_input("URL Articol:", placeholder="Paste link here...", key="url_input")
        if url:
            try:
                config = Config()
                config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
                
                article = Article(url, config=config)
                article.download()
                article.parse()
                
                titlu_analiza = article.title
                text_analiza = article.text
                
                if titlu_analiza:
                    st.success(f"Articol detectat: {titlu_analiza}")
                else:
                    st.warning("Nu am putut extrage titlul. Introdu-l manual la tab-ul 'Text Manual'.")
            except Exception as e:
                st.error(f"Eroare la accesarea site-ului: {e}")

    with input_mode[1]:
        titlu_analiza = st.text_area("Titlu / Paragraf:", height=100, key="manual_text")
    
    c1, c2 = st.columns([1, 5])
    with c1:
        # Folosim un buton simplu pentru a declanșa analiza
        start_analysis = st.button("Analizează", type="primary", use_container_width=True)
    with c2:
        if st.button("Reset", type="secondary"):
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. RESULTS ---
if (start_analysis or titlu_analiza) and titlu_analiza:
    res_titlu = analyze_text(titlu_analiza)
    
    if res_titlu:
        st.markdown(f"""
            <div style="background: white; border: 1px solid #e2e8f0; padding: 25px; border-radius: 12px; border-top: 5px solid {res_titlu['config']['color']};">
                <div class="verdict-badge" style="background-color: {res_titlu['config']['color']};">
                    {res_titlu['label']}
                </div>
                <h3 style="margin-top: 0; color: #0f172a;">{titlu_analiza}</h3>
                <p style="color: #64748b; font-size: 15px;">{res_titlu['config']['desc']}</p>
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px;">
                    <span style="font-size: 13px; font-weight: 600; color: #94a3b8;">ÎNCREDERE MODEL:</span>
                    <span style="font-size: 13px; font-weight: 700; color: {res_titlu['config']['color']};">{res_titlu['score']:.2%}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if text_analiza:
            st.markdown("---")
            res_content = analyze_text(text_analiza)
            
            col1, col2 = st.columns(2)
            col1.metric("Verdict Titlu", res_titlu['label'])
            col2.metric("Verdict Conținut", res_content['label'])
            
            if res_titlu['label'] != res_content['label']:
                st.warning(f"S-a detectat o diferență de ton între titlu ({res_titlu['label']}) și conținut ({res_content['label']}).")

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("SenticGuard v12")
    st.markdown("---")
    for cat, info in CATEGORIES.items():
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <span style="color:{info['color']}; font-weight:bold;">{cat}</span><br>
            <small style="color:#64748b;">{info['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
