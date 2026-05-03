# SenticGuard: AI-Powered Emotional Integrity & Media Resilience Framework  
### SenticGuard is an advanced intelligent system designed to identify emotional manipulation, sensationalism, and disinformation patterns in Romanian news feeds using NLP technology.  

**The Problem: Psychological Manipulation**  
Modern media often bypasses rational filters by using psychological triggers—fear, false urgency, or conflict—to drive engagement. This project aims to map these nuances to reduce social anxiety and improve digital media literacy.

**The Solution: Human-in-the-Loop Active Learning**  
Unlike static filters, SenticGuard employs a continuous improvement cycle. The AI proposes classifications, and a specialist validates them, creating a dataset that constantly refines the model's accuracy.

**Multidimensional Classification (6 Categories)**  
The system has evolved from simple binary detection to a nuanced 6-category mapping:

OBJECTIVE: Factual, neutral reporting.

ALARMIST: Headlines inducing panic or exaggerated fear.

SENZATIONAL: Clickbait designed to exploit curiosity.

CONFLICTUAL: Focus on scandals, accusations, and disputes.

INFORMATIVE: Utilitarian content (weather, public service announcements).

OPINION: Subjective editorials and personal analyses.

**Workflow & Infrastructure**  

* Collection & Filtering (Admin App): Headlines are retrieved via RSS feeds. The system allows for selective labeling to maintain a high-quality, balanced dataset.

* Validation: Real-time AI confidence scores (BERT-based) assist the admin in confirming or correcting labels.

* Storage: Data is synchronized via Google Sheets API, serving as a dynamic, version-controlled dataset.

* Training: Fine-tuning is performed on BERT/RoBERTa architectures using specialized GPU environments to handle Romanian linguistic nuances.

* Deployment: The public-facing app automatically utilizes the latest model weights, providing real-time manipulation analysis for the end user. Demo (romanian) [here](https://senticguard-app.streamlit.app/)

**Technology Stack**

* AI/NLP: transformers (BERT/RoBERTa), PyTorch, pipeline API.

* Frontend: Streamlit (Admin & Public Verifier).

* Data Pipeline: GSheetsConnection for seamless dataset management.

* Automation: gdown for automated model weight synchronization from Google Drive.

**Vision**  
While currently optimized for the Romanian media landscape, the architecture is being refined to support multilingual analysis (English/XLM-RoBERTa), aiming to provide a cross-border shield against global disinformation patterns.
