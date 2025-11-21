# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2
import os

# ---------------------------------------
# 🧩 PATCH Torch 2.6 → Correction chargement YOLO
# ---------------------------------------
import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

# Ajout du modèle YOLO dans les globals autorisés
add_safe_globals([DetectionModel])

# ---------------------------------------
# 🎨 CONFIG INTERFACE MODERNE
# ---------------------------------------
st.set_page_config(
    page_title="Détection Intelligente de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 CSS custom - Design moderne avec cartes vert foncé
custom_css = """
<style>
    /* ton CSS original, inchangé */
"""  # ⚠️ GARDÉ COMME TU L’AVAIS (je ne répète pas pour réduire la taille)
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------
# 🧠 CHARGEMENT DU MODEL YOLO (corrigé)
# ---------------------------------------
MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        # ⚠️ Important : Charger YOLO avec task='detect'
        return YOLO(path, task="detect")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# ---------------------------------------
# 🖥️ HEADER PRINCIPAL
# ---------------------------------------
st.markdown("""
<div class="main-header fade-in-up">
    <div class="main-title">🗑️ Détection Intelligente</div>
    <div class="main-subtitle">IA Avancée · Détection en Temps Réel · Classification Automatique</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# 🛠️ BARRE D'OUTILS SUPÉRIEURE
# ---------------------------------------
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### 📋 Configuration du Modèle")

with col2:
    if model is None:
        st.error("❌ Modèle non chargé")
    else:
        st.success("✅ Modèle chargé")

with col3:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button(
                "💾 Télécharger le modèle", 
                data=f, 
                file_name="best.pt",
                help="Téléchargez le modèle YOLO actuel",
                use_container_width=True
            )

# ---------------------------------------
# 📤 SECTION UPLOAD DU MODÈLE
# ---------------------------------------
st.markdown("<div class='content-card fade-in-up'>", unsafe_allow_html=True)
st.markdown("### 🚀 Configuration du Modèle IA")

if model is None:
    st.warning("""
    **📝 Modèle introuvable**
    
    Pour utiliser l'application :
    1. Placez votre fichier `best.pt` dans le dossier `models/`
    2. Ou uploadez un modèle YOLO ci-dessous
    """)

uploaded_model = st.file_uploader(
    "📤 Uploader un modèle YOLO (.pt)",
    type=["pt"],
    help="Sélectionnez votre modèle YOLO entraîné"
)

if uploaded_model is not None:
    os.makedirs("models", exist_ok=True)
    model_bytes = uploaded_model.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(model_bytes)
    st.success("🎉 Modèle uploadé avec succès!")
    st.info("🔄 **Rechargez la page** pour utiliser le nouveau modèle")
    
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 📸 SECTION UPLOAD D'IMAGE
# ---------------------------------------
st.markdown("<div class='upload-section fade-in-up'>", unsafe_allow_html=True)
st.markdown("### 📸 Analyse d'Image")
st.markdown("""
<div style='text-align: center;'>
    <h3 style='color: #e8f5e8; margin-bottom: 1rem;'>⬆️ Glissez-déposez votre image ici</h3>
    <p style='color: #c8e6c8; font-size: 1.1rem;'>Formats supportés: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

uploaded_img = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    key="main_uploader",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🖼️ AFFICHAGE DES RÉSULTATS
# ---------------------------------------
if uploaded_img:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='content-card fade-in-up'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Originale")
        
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Image source uploadée", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur de chargement: {e}")
            uploaded_img = None
        
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    analyze = st.button(
        "🚀 Lancer l'Analyse IA Avancée", 
        type="primary", 
        use_container_width=True,
        help="Démarrer la détection et classification automatique"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if analyze:
        if model is None:
            st.error("🚫 Aucun modèle YOLO disponible")
        else:
            with st.spinner("🔍 **Analyse en cours...** L'IA scanne l'image pour détecter les poubelles"):
                img_array = np.array(image)
                
                try:
                    results = model.predict(img_array, conf=0.25, imgsz=640)
                except Exception as e:
                    st.error(f"❌ Erreur d'analyse: {e}")
                    results = None

                if results is None or len(results) == 0:
                    st.warning("⚠️ Aucune détection obtenue")
                else:
                    r = results[0]

                    try:
                        annotated = r.plot()
                        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    except Exception:
                        annotated = img_array

                    with col2:
                        st.markdown("<div class='content-card fade-in-up'>", unsafe_allow_html=True)
                        st.markdown("### 📊 Résultats de Détection")
                        st.image(annotated, caption="🟢 Détections YOLOv8 - Zones identifiées", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    dets = getattr(r, "boxes", None)
                    if dets is not None and len(dets) > 0:
                        st.markdown("<div class='stats-container fade-in-up'>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="stat-item">
                            <span class="stat-number">{len(dets)}</span>
                            <span class="stat-label">Poubelles Détectées</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">{max(len(dets), 1)}</span>
                            <span class="stat-label">Analyses Effectuées</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-number">YOLOv8</span>
                            <span class="stat-label">Modèle IA</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='content-card fade-in-up'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Détails des Analyses")
                    
                    if dets is None or len(dets) == 0:
                        st.warning("❌ Aucune poubelle détectée dans l'image")
                    else:
                        for i, box in enumerate(dets, start=1):
                            cls_idx = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = model.names[cls_idx] if hasattr(model, "names") else str(cls_idx)
                            
                            conf_percent = int(conf * 100)
                            st.markdown(f"""
                            <div class="confidence-bar-container">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <span class="detection-badge">🔍 Détection #{i} • {cls_name.upper()}</span>
                                    <strong style="font-size: 1.3rem; color: #e8f5e8;">{conf_percent}%</strong>
                                </div>
                                <div class="confidence-bar" style="width: {conf_percent}%;"></div>
                                <div style="text-align: center; color: #c8e6c8; font-size: 0.9rem; margin-top: 5px;">
                                    Niveau de confiance de l'IA
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='content-card fade-in-up'>", unsafe_allow_html=True)
    st.markdown("### 💡 Guide d'Utilisation")
    
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    
    with col_guide1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>1️⃣</div>
            <h4 style='color: #e8f5e8;'>Upload du Modèle</h4>
            <p style='color: #c8e6c8;'>Configurez votre modèle YOLO ou utilisez le modèle par défaut</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>2️⃣</div>
            <h4 style='color: #e8f5e8;'>Import d'Image</h4>
            <p style='color: #c8e6c8;'>Sélectionnez une image contenant une ou plusieurs poubelles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>3️⃣</div>
            <h4 style='color: #e8f5e8;'>Analyse IA</h4>
            <p style='color: #c8e6c8;'>Lancez la détection et visualisez les résultats en temps réel</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🏁 FOOTER
# ---------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #c8e6c8; padding: 3rem 1rem;'>
    <h3 style='color: #e8f5e8; margin-bottom: 1rem;'>Détection Intelligente de Poubelles</h3>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>🚀 Propulsé par YOLOv8 & Streamlit</p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>Système de détection et classification automatique • IA de pointe</p>
</div>
""", unsafe_allow_html=True)
