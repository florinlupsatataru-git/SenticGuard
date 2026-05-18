import streamlit as st
import feedparser
import pandas as pd
import gdown
import os
import psycopg2
from dotenv import load_dotenv
import plotly.express as px
from streamlit_gsheets import GSheetsConnection
from transformers import pipeline

# --- 0. ENVIRONMENT SETUP ---
# Load environment variables from .env file (local on server, not on GitHub)
load_dotenv()

# --- 1. CATEGORIES CONFIG ---
CATEGORIES_MAP = {
    "OBIECTIV": 0,
    "ALARMIST": 1,
    "SENZATIONAL": 2,
    "CONFLICTUAL": 3,
    "INFORMATIV": 4,
    "OPINIE": 5
}
CATEGORII_LIST = ["NU ETICHETA"] + list(CATEGORIES_MAP.keys())

# --- 2. LOGIN AND SECURITY ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def get_db_connection():
    """Establish connection to PostgreSQL using environment variables"""
    return psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME")
    )

def login():
    st.title("🛡️ SenticGuard Admin Panel")
    parola = st.text_input("Introdu parola de administrator", type="password")
    if st.button("Log In"):
        # Check against ADMIN_PASSWORD from .env
        admin_pass = os.getenv("ADMIN_PASSWORD")
        if parola == admin_pass:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Parolă incorectă")

if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- 3. DATA LOADING FUNCTIONS ---
def load_global_data():
    """Load the main training dataset from GSheets"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        st.session_state.df = conn.read(ttl=0) 
    except Exception as e:
        st.error(f"Error loading training data: {e}")

def load_logs():
    """Load the analysis logs from the 'Logs' worksheet"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        return conn.read(worksheet="Logs", ttl=0)
    except Exception as e:
        st.error(f"Error loading logs: {e}")
        return pd.DataFrame()

def load_security_logs():
    """Load security access logs from PostgreSQL"""
    try:
        conn = get_db_connection()
        query = "SELECT timestamp, ip_address, event_type, severity, description FROM security_logs ORDER BY timestamp DESC LIMIT 200"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

def load_pending_queue():
    """Fetch all unvalidated articles aggregated by the background crawler"""
    try:
        conn = get_db_connection()
        query = "SELECT id, title, content, predicted_label, confidence_score FROM pending_queue ORDER BY fetched_at ASC"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading automated queue: {e}")
        return pd.DataFrame()

# Initial data load
if "df" not in st.session_state:
    load_global_data()

# --- 4. MODEL LOADING ---
@st.cache_resource
def load_classifier():
    """Download and initialize the local classifier for suggestion purposes"""
    save_path = "./model_temp"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    files = {
        "config.json": "1kKvuBeOMMiIXF8_UtP3zoI6lxi6RPvo8",
        "model.safetensors": "1Nrnp84om-EKrNDDdeS8JcYTArwOwusF7", 
        "tokenizer_config.json": "1If1PxDO6I9-ch55fgqxqFl9X0AHtWwjY",
        "tokenizer.json": "1e1WkWXewonur2dTuPekixV-liMQhigQi"
    }
    
    try:
        for name, file_id in files.items():
            local_file = os.path.join(save_path, name)
            if not os.path.exists(local_file):
                file_url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(file_url, local_file, quiet=True)
        return pipeline("text-classification", model=save_path, tokenizer=save_path)
    except Exception as e:
        st.error(f"Eroare la descărcarea modelului: {e}")
        return None

# --- 5. MAIN NAVIGATION (TABS) ---
st.title("🛡️ SenticGuard Master Control")
tab_stats, tab_training, tab_auto_validation, tab_security = st.tabs([
    "📈 Statistici Live", 
    "📊 Colectare & Training", 
    "🤖 Validare Automatizată",
    "🛡️ Monitorizare Securitate"
])

# ==========================================
# TAB 1: LIVE STATISTICS (FROM GOOGLE LOGS)
# ==========================================
with tab_stats:
    col_title, col_refresh = st.columns([0.8, 0.2])
    with col_title:
        st.header("Monitorizare Verificări Utilizatori")
    with col_refresh:
        if st.button("🔄", key="refresh_stats"):
            st.rerun()

    df_logs = load_logs()
    
    if not df_logs.empty:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Verificări", len(df_logs))
        top_src = df_logs['source'].mode()[0] if not df_logs['source'].empty else "N/A"
        m2.metric("Site de Top", top_src)
        m3.metric("Confidență Medie", f"{df_logs['confidence'].mean():.1%}")
        
        st.divider()
        col_pie, col_bar = st.columns(2)
        
        with col_pie:
            st.subheader("⚖️ Distribuție Verdicte")
            fig_verdicte = px.pie(df_logs, names='verdict', color='verdict',
                                 color_discrete_map={
                                     "OBIECTIV": "#10b981", "ALARMIST": "#ef4444", 
                                     "SENZATIONAL": "#f59e0b", "CONFLICTUAL": "#8b5cf6",
                                     "INFORMATIV": "#3b82f6", "OPINIE": "#64748b"
                                 })
            st.plotly_chart(fig_verdicte, use_container_width=True)
            
        with col_bar:
            st.subheader("Top 5 Surse Verificate")
            top_sources = df_logs['source'].value_counts().head(5).reset_index()
            fig_sources = px.bar(top_sources, x='count', y='source', orientation='h')
            st.plotly_chart(fig_sources, use_container_width=True)

        st.subheader("Detalii Analize Recente")
        st.dataframe(df_logs.sort_values(by='timestamp', ascending=False), use_container_width=True)
    else:
        st.info("Încă nu există date în foaia de Logs.")

# ==========================================
# TAB 2: COLLECTION & TRAINING (MANUAL)
# ==========================================
with tab_training:
    st.header("Colectare & Validare Date")
    
    RSS_FEEDS = {
        "Mediafax": "https://www.mediafax.ro/rss",
        "Hotnews": "https://www.hotnews.ro/rss",
        "Digi24": "https://www.digi24.ro/rss",
        "G4Media": "https://www.g4media.ro/feed",
        "Libertatea": "https://www.libertatea.ro/feed",
        "Stirile ProTV": "https://stirileprotv.ro/rss",
        "Antena 3": "https://www.antena3.ro/rss"
    }

    sursa = st.selectbox("Alege sursa de știri:", list(RSS_FEEDS.keys()))

    if st.button("Aduceți titluri noi"):
        with st.spinner('AI-ul clasifică știrile...'):
            feed = feedparser.parse(RSS_FEEDS[sursa])
            classifier = load_classifier()
            new_data = []
            for entry in feed.entries[:30]:
                label_sugerat, scor_ai = "OBIECTIV", 0.0
                if classifier:
                    rezultat = classifier(entry.title)[0]
                    label_sugerat, scor_ai = rezultat['label'], rezultat['score']
                new_data.append({"text": entry.title, "ai_label": label_sugerat, "ai_score": scor_ai})
            st.session_state.temp_df = pd.DataFrame(new_data)

    if "temp_df" in st.session_state:
        valid_entries = []
        for index, row in st.session_state.temp_df.iterrows():
            st.markdown(f"**{index+1}.** {row['text']}")
            col_select, col_score = st.columns([0.4, 0.6])
            with col_select:
                idx_def = (list(CATEGORIES_MAP.keys()).index(row['ai_label']) + 1) if row['ai_label'] in CATEGORIES_MAP else 0
                alegere = st.selectbox(f"Label {index}", options=CATEGORII_LIST, index=idx_def, key=f"sel_{index}")
            with col_score:
                st.write(f"**AI:** {row['ai_label']} ({row['ai_score']:.1%})")
            if alegere != "NU ETICHETA":
                valid_entries.append({"text": row['text'], "label": CATEGORIES_MAP[alegere]})
            st.divider()

        if st.button("Confirmă și Salvează în Dataset", type="primary"):
            if valid_entries:
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    existing_df = conn.read()
                    updated_df = pd.concat([existing_df, pd.DataFrame(valid_entries)], ignore_index=True)
                    conn.update(data=updated_df)
                    st.success("Date salvate cu succes!")
                    del st.session_state.temp_df
                    st.rerun()
                except Exception as e:
                    st.error(f"Eroare: {e}")

# ==========================================
# TAB 3: AUTOMATED VALIDATION (SMART QUEUE)
# ==========================================
with tab_auto_validation:
    st.header("🤖 Validare Coadă Automatizată")
    st.write("Aici se află articolele colectate automat în fundal de către crawler-ul de știri.")

    if st.button("🔄 Refresh Queue", key="refresh_queue"):
        st.rerun()

    df_queue = load_pending_queue()

    if df_queue.empty:
        st.success("Coada este goală! Nu există articole noi de validat.")
    else:
        st.info(f"Există **{len(df_queue)}** articole care așteaptă decizia ta.")
        
        # Lists to keep tracking of records to update/delete on final submission
        final_validation_list = []
        ids_to_delete = []

        for index, row in df_queue.iterrows():
            st.markdown(f"**{index+1}.** {row['title']}")
            
            col_select, col_score = st.columns([0.4, 0.6])
            with col_select:
                # Pre-select the dropdown option based on the crawler's initial AI prediction
                idx_def = (list(CATEGORIES_MAP.keys()).index(row['predicted_label']) + 1) if row['predicted_label'] in CATEGORIES_MAP else 0
                alegere = st.selectbox(
                    f"Verdict final pentru articolul #{row['id']}", 
                    options=CATEGORII_LIST, 
                    index=idx_def, 
                    key=f"auto_sel_{row['id']}"
                )
            with col_score:
                st.write(f"**AI Prediction:** {row['predicted_label']} ({row['confidence_score']:.1%})")
            
            # Map choice to final destination tracking
            if alegere != "NU ETICHETA":
                final_validation_list.append({"text": row['title'], "label": CATEGORIES_MAP[alegere], "db_id": row['id']})
            else:
                # If marked as "NU ETICHETA", it means the user wants to discard/delete it completely
                ids_to_delete.append(row['id'])
                
            st.divider()

        # Submit workflow process
        if st.button("Procesează coada de articole", type="primary", key="btn_submit_queue"):
            try:
                # 1. Save validated entries to Google Sheets if any exist
                if final_validation_list:
                    conn_gsheets = st.connection("gsheets", type=GSheetsConnection)
                    existing_df = conn_gsheets.read()
                    
                    # Remove the internal tracking 'db_id' before saving to sheets
                    clean_entries = [{"text": x["text"], "label": x["label"]} for x in final_validation_list]
                    
                    updated_df = pd.concat([existing_df, pd.DataFrame(clean_entries)], ignore_index=True)
                    conn_gsheets.update(data=updated_df)
                    
                    # Add database IDs of validated items to the master deletion list
                    for item in final_validation_list:
                        ids_to_delete.append(item["db_id"])

                # 2. Delete processed/discarded items from local PostgreSQL pending_queue
                if ids_to_delete:
                    conn_pg = get_db_connection()
                    cur = conn_pg.cursor()
                    # Execute bulk deletion query
                    cur.execute("DELETE FROM pending_queue WHERE id = ANY(%s);", (ids_to_delete,))
                    conn_pg.commit()
                    cur.close()
                    conn_pg.close()

                st.success("Coada a fost procesată cu succes! Datele au fost mutate în Google Sheets.")
                st.rerun()

            except Exception as e:
                st.error(f"Eroare la procesarea cozii de date: {e}")

# ==========================================
# TAB 4: SECURITY LOGS (FROM POSTGRESQL)
# ==========================================
with tab_security:
    st.header("🛡️ Monitorizare Securitate și Acces")
    st.write("Aceste log-uri sunt preluate în timp real din baza de date PostgreSQL locală.")
    
    if st.button("🔄 Refresh Security Logs", key="refresh_sec"):
        st.rerun()
    
    sec_logs = load_security_logs()
    
    if not sec_logs.empty:
        st.dataframe(sec_logs, use_container_width=True)
    else:
        st.warning("Nu s-au găsit log-uri de securitate.")

# ==========================================
# SIDEBAR: DATASET SUMMARY & MANUAL ADD
# ==========================================
st.sidebar.title("📖 Legenda Categoriilor")

if "df" in st.session_state and not st.session_state.df.empty:
    counts = st.session_state.df['label'].value_counts()
    st.sidebar.info(f"""
    - **0. OBIECTIV ({int(counts.get(0, 0))}):** Neutru.
    - **1. ALARMIST ({int(counts.get(1, 0))}):** Panică.
    - **2. SENZAȚIONAL ({int(counts.get(2, 0))}):** Clickbait.
    - **3. CONFLICTUAL ({int(counts.get(3, 0))}):** Scandal.
    - **4. INFORMATIV ({int(counts.get(4, 0))}):** Utilitar.
    - **5. OPINIE ({int(counts.get(5, 0))}):** Editoriale.
    ---
    **Total Dataset: {len(st.session_state.df)}**
    """)

st.sidebar.divider()
st.sidebar.title("Adăugare Manuală")
with st.sidebar.form("manual_add_form", clear_on_submit=True):
    manual_text = st.text_area("Titlu știre nouă:")
    manual_cat = st.selectbox("Categorie:", list(CATEGORIES_MAP.keys()))
    submitted = st.form_submit_button("Salvează Titlu")

    if submitted:
        if manual_text:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                existing_df = conn.read()
                new_row = pd.DataFrame({"text": [manual_text], "label": [CATEGORIES_MAP[manual_cat]]})
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                conn.update(data=updated_df)
                st.sidebar.success("Adăugat!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Eroare: {e}")
