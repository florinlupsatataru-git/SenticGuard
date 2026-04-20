import streamlit as st
import feedparser
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from transformers import pipeline

# 1. Login and security
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

# 2. Model loading for prediction
#@st.cache_resource
#def load_classifier():
#    model_path = "/content/drive/MyDrive/NewsAnalyzer/model_alarmism_final"
#    try:
#        return pipeline("text-classification", model=model_path, tokenizer=model_path)
#    except:
#        return None

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
            if not os.path.exists(local_file): # Descarcă doar dacă nu există deja
                file_url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(file_url, local_file, quiet=True)
        
        return pipeline("text-classification", model=save_path, tokenizer=save_path)
    except Exception as e:
        st.error(f"Eroare la descărcarea modelului: {e}")
        return None


# 3. Admin user interface
st.title("🚀 Colectare & Validare AI")

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
    with st.spinner('AI-ul analizează titlurile proaspete...'):
        feed = feedparser.parse(RSS_FEEDS[sursa])
        classifier = load_classifier()
        
        new_data = []
        for entry in feed.entries[:30]:
            ai_suggested = False
            if classifier:
                prediction = classifier(entry.title)[0]
                # Ajustează LABEL_1 în funcție de cum a mapat modelul tău (de obicei LABEL_1 e pozitiv/alarmist)
                ai_suggested = True if prediction['label'] == 'LABEL_1' else False
            
            new_data.append({
                "text": entry.title,
                "ai_suggested": ai_suggested
            })
        
        st.session_state.temp_df = pd.DataFrame(new_data)

if "temp_df" in st.session_state:
    # Calculăm statistici rapide
    nr_alarmiste = st.session_state.temp_df["ai_suggested"].sum()
    st.metric("Detecție AI", f"{nr_alarmiste} știri alarmiste", f"{len(st.session_state.temp_df)} total")
    
    st.info("💡 AI-ul a bifat automat titlurile suspecte. Corectează manual unde este cazul!")

    updated_labels = []
    for index, row in st.session_state.temp_df.iterrows():
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            # Checkbox-ul pornește bifat dacă ai_suggested e True
            is_alarmist = st.checkbox("Alarmist", value=row["ai_suggested"], key=f"check_{index}")
        with col2:
            if row["ai_suggested"]:
                st.markdown(f"🚩 **{row['text']}**")
            else:
                st.write(row["text"])
        
        updated_labels.append(1 if is_alarmist else 0)

    if st.button("💾 Confirmă și Salvează în Google Sheets"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            existing_df = conn.read()
            
            to_save = pd.DataFrame({
                "text": st.session_state.temp_df["text"],
                "label": updated_labels
            })
            
            updated_df = pd.concat([existing_df, to_save], ignore_index=True)
            conn.update(data=updated_df)
            
            st.success("✅ Datele au fost validate și salvate!")
            del st.session_state.temp_df
        except Exception as e:
            st.error(f"Eroare la salvare: {e}")
