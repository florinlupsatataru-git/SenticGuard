import streamlit as st
import feedparser
import pandas as pd
import gdown
import os
from streamlit_gsheets import GSheetsConnection
from transformers import pipeline

# --- 1. CATEGORIES CONFIG ---
CATEGORIES_MAP = {
    "OBIECTIV": 0,
    "ALARMIST": 1,
    "SENZATIONAL": 2,
    "CONFLICTUAL": 3,
    "INFORMATIV": 4,
    "OPINIE": 5
}
# Added "NU ETICHETA" to allow skipping irrelevant news
CATEGORII_LIST = ["NU ETICHETA"] + list(CATEGORIES_MAP.keys())

# --- 2. LOGIN AND SECURITY ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.title("🔐 SenticGuard Admin Panel")
    parola = st.text_input("Introdu parola de administrator", type="password")
    if st.button("Log In"):
        if parola == st.secrets["ADMIN_PASSWORD"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Parolă incorectă")

if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- 2.5 LOAD DATASET GLOBALLY ---
def load_global_data():
    try:
        # ttl=0 forcing read new data, no cache
        conn = st.connection("gsheets", type=GSheetsConnection)
        st.session_state.df = conn.read(ttl=0) 
    except Exception as e:
        st.error(f"Error: {e}")

# Load data immediately after authentication
if "df" not in st.session_state:
    load_global_data()

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_classifier():
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

# --- 4. ADMIN USER INTERFACE ---
st.title("🚀 Colectare & Validare AI (v2.0 - 6 Categorii)")

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
    with st.spinner('AI-ul clasifică știrile pe cele 6 categorii...'):
        feed = feedparser.parse(RSS_FEEDS[sursa])
        classifier = load_classifier()
        
        new_data = []
        for entry in feed.entries[:30]:
            label_sugerat = "OBIECTIV"
            scor_ai = 0.0
            if classifier:
                rezultat = classifier(entry.title)[0]
                label_sugerat = rezultat['label']
                scor_ai = rezultat['score']
            new_data.append({"text": entry.title, "ai_label": label_sugerat, "ai_score": scor_ai})
        st.session_state.temp_df = pd.DataFrame(new_data)


# --- 5. VALIDATION ---
if "temp_df" in st.session_state:
    st.write("### 📝 Analiză și Validare Manuală")
    
    # Logic to collect only labeled news
    valid_entries = []
    
    for index, row in st.session_state.temp_df.iterrows():
        st.markdown(f"**{index+1}.** {row['text']}")
        
        col_select, col_score = st.columns([0.4, 0.6])
        
        with col_select:
            # Shift AI index by +1 because "NU ETICHETA" is now at index 0
            index_default = (list(CATEGORIES_MAP.keys()).index(row['ai_label']) + 1) if row['ai_label'] in CATEGORIES_MAP else 0

            alegere = st.selectbox(
                   f"Label {index}", 
                   options=CATEGORII_LIST,
                   index=index_default,
                   key=f"{sursa}_select_{index}", 
                   label_visibility="collapsed"
            )
            
        with col_score:
            conf_color = "🟢" if row['ai_score'] > 0.8 else "🟡" if row['ai_score'] > 0.6 else "🔴"
            st.write(f"{conf_color} **AI:** {row['ai_label']} ({row['ai_score']:.1%})")
        
        # Only add to list if not skipped
        if alegere != "NU ETICHETA":
            valid_entries.append({"text": row['text'], "label": CATEGORIES_MAP[alegere]})
            
        st.divider()

    if st.button("💾 Confirmă și Salvează în Dataset", type="primary"):
        if not valid_entries:
            st.warning("⚠️ Nu ai selectat niciun titlu pentru validare (toate sunt setate pe 'NU ETICHETA').")
        else:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                existing_df = conn.read()
                
                # Convert list of valid entries to DataFrame
                to_save = pd.DataFrame(valid_entries)
                
                updated_df = pd.concat([existing_df, to_save], ignore_index=True)
                conn.update(data=updated_df)
                
                # Update local session state for sidebar consistency
                st.session_state.df = updated_df
                
                st.success(f"✅ Am adăugat {len(to_save)} titluri noi! Cele ignorate au fost filtrate.")
                del st.session_state.temp_df
                st.rerun()
            except Exception as e:
                st.error(f"Eroare la salvare: {e}")

# --- SIDEBAR: LEGEND OF CATEGORIES WITH STATISTICS ---

# --- CALCULATION OF LABEL DISTRIBUTION ---
counts = {i: 0 for i in range(6)}

if "df" in st.session_state and st.session_state.df is not None:
    temp_df_counts = st.session_state.df.copy()
    temp_df_counts['label'] = pd.to_numeric(temp_df_counts['label'], errors='coerce')
    real_counts = temp_df_counts['label'].value_counts().to_dict()
    for k, v in real_counts.items():
        if k in counts:
            counts[int(k)] = v

if st.sidebar.button("🔄 Refresh Data"):
    load_global_data()
    st.rerun()

st.sidebar.title("📖 Legenda Categoriilor")

with st.sidebar.expander("Vezi descrierea și statisticile", expanded=True):
     st.markdown(f"""
          - **0. OBIECTIV ({int(counts.get(0, 0))}):** Știri neutre, bazate pe fapte verificate, fără nuanțe emoționale.
          - **1. ALARMIST ({int(counts.get(1, 0))}):** Titluri care induc panică, teamă sau folosesc avertismente exagerate.
          - **2. SENZAȚIONAL ({int(counts.get(2, 0))}):** Clickbait pur, mizează pe curiozitate sau șoc (ex: "Nu o să crezi").
          - **3. CONFLICTUAL ({int(counts.get(3, 0))}):** Scandaluri, certuri, acuzații directe sau dispute între persoane/grupuri.
          - **4. INFORMATIV ({int(counts.get(4, 0))}):** Conținut utilitar, ghiduri, prognoze meteo concrete sau anunțuri de interes public.
          - **5. OPINIE ({int(counts.get(5, 0))}):** Editoriale, comentarii subiective sau analize semnate de autori.
          ---
          **Total în dataset: {len(st.session_state.df) if "df" in st.session_state else 0} înregistrări**
     """)

# --- SIDEBAR: MANUAL ADD ---
st.sidebar.divider()
st.sidebar.title("➕ Adăugare Manuală")

with st.sidebar.form("manual_add_form", clear_on_submit=True):
    manual_text = st.text_area("Titlu știre nouă:", key="manual_text_input")
    # Exclude "NU ETICHETA" from manual add because manual entry implies intent to label
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
                st.session_state.df = updated_df
                st.success("✅ Titlu salvat!")
            except Exception as e:
                st.error(f"Eroare: {e}")
        else:
            st.warning("Te rog să introduci un titlu.")
