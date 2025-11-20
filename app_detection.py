# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import cv2
import os

# ---------------------------------------
# 🎨 CONFIG INTERFACE AMÉLIORÉE
# ---------------------------------------
st.set_page_config(
    page_title="Détection Intelligente de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 CSS custom élégant avec dégradés et animations
custom_css = """
<style>
    /* Thème principal */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Container principal */
    .main .block-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* En-tête stylisé */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Boutons modernes */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Cartes de contenu */
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar stylisée (sélecteur stable) */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #eef3ff 0%, #e3e9f9 50%, #d9e2f7 100%);
        box-shadow: 4px 0 15px -5px rgba(0,0,0,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #1f2d4d !important;
    }
    /* Accent sur les titres dans la sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] .stMarkdown h2 {
        color: #445b8a !important;
        text-shadow: none;
    }
    /* Uniformiser fond des containers internes */
    [data-testid="stSidebar"] .stButton>button, [data-testid="stSidebar"] .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stFileUploader, [data-testid="stSidebar"] .stDownloadButton, [data-testid="stSidebar"] .stMarkdown {
        background: transparent !important;
    }
    
    /* Upload area améliorée */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(102, 126, 234, 0.1);
        border-color: #764ba2;
    }
    
    /* Badges de résultats */
    .detection-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        margin: 5px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 176, 155, 0.3);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
        height: 8px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------
# 🧠 CHARGEMENT DU MODEL YOLO (inchangé)
# ---------------------------------------
MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# ---------------------------------------
# 🖥️ HEADER INTERFACE AMÉLIORÉ
# ---------------------------------------
st.markdown("""
<div class="main-header fade-in">
    <div class="main-title">🗑️ Détection Intelligente de Poubelles</div>
    <div class="main-subtitle">IA Avancée · YOLOv8 · Classification Automatique (Vide/Pleine)</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------
# SIDEBAR: Model actions (design amélioré)
# ---------------------------------------
with st.sidebar:
    st.markdown("""
    <div style='color: white; text-align: center; padding: 1rem;'>
        <h2>🛠️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.warning("📋 Modèle introuvable")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: white;'>
            Placez <code>best.pt</code> dans le dossier <code>models/</code> ou uploadez ci-dessous.
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_model = st.file_uploader("📤 Uploader un modèle YOLO", type=["pt"], help="Sélectionnez votre fichier .pt")
    
    if uploaded_model is not None:
        os.makedirs("models", exist_ok=True)
        model_bytes = uploaded_model.read()
        with open(MODEL_PATH, "wb") as f:
            f.write(model_bytes)
        st.success("✅ Modèle uploadé avec succès!")
        st.info("🔄 Rechargez la page pour l'utiliser")
    
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button(
                "💾 Télécharger le modèle", 
                data=f, 
                file_name="best.pt",
                help="Téléchargez le modèle actuel best.pt"
            )

# ---------------------------------------
# 📤 UPLOAD D'IMAGE (design amélioré)
# ---------------------------------------
st.markdown("<div class='content-card fade-in'>", unsafe_allow_html=True)
st.markdown("### 📸 Upload d'Image")

uploaded_img = st.file_uploader(
    "Glissez-déposez ou sélectionnez une image de poubelle",
    type=["jpg", "jpeg", "png"],
    key="uploader"
)

st.markdown("</div>", unsafe_allow_html=True)

# Affichage des images côte à côte
col1, col2 = st.columns([1, 1])

with col1:
    if uploaded_img:
        st.markdown("<div class='content-card fade-in'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Originale")
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Image source", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur: {e}")
            uploaded_img = None
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🕹️ BOUTON ANALYSE + LOGIQUE YOLO (amélioré)
# ---------------------------------------
if uploaded_img:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    analyze = st.button(
        "🔍 Lancer l'Analyse IA", 
        type="primary", 
        use_container_width=True,
        help="Démarrer la détection et classification des poubelles"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if analyze:
        if model is None:
            st.error("🚫 Modèle YOLO non disponible")
        else:
            with st.spinner("🔄 Analyse en cours... L'IA scanne l'image"):
                # Conversion et prédiction
                img_array = np.array(image)
                
                try:
                    results = model.predict(img_array, conf=0.25, imgsz=640)
                except Exception as e:
                    st.error(f"❌ Erreur d'analyse: {e}")
                    results = None

                if results is None or len(results) == 0:
                    st.warning("⚠️ Aucun résultat obtenu")
                else:
                    r = results[0]

                    # Image annotée
                    try:
                        annotated = r.plot()
                        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    except Exception:
                        annotated = img_array

                    # Affichage résultats
                    with col2:
                        st.markdown("<div class='content-card fade-in'>", unsafe_allow_html=True)
                        st.markdown("### 📊 Résultats de l'Analyse")
                        st.image(annotated, caption="Détections YOLOv8", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Détails des détections
                    st.markdown("<div class='content-card fade-in'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Détails des Détections")
                    
                    dets = getattr(r, "boxes", None)
                    if dets is None or len(dets) == 0:
                        st.warning("❌ Aucune poubelle détectée")
                    else:
                        for i, box in enumerate(dets, start=1):
                            cls_idx = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = model.names[cls_idx] if hasattr(model, "names") else str(cls_idx)
                            
                            # Barre de confiance visuelle
                            conf_percent = int(conf * 100)
                            st.markdown(f"""
                            <div style='margin: 1rem 0; padding: 1rem; background: #f8f9fa; border-radius: 10px;'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <span class='detection-badge'>#{i} • {cls_name}</span>
                                    <strong>{conf_percent}%</strong>
                                </div>
                                <div class='confidence-bar' style='width: {conf_percent}%;'></div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.info("""
    ### 💡 Comment utiliser cette application:
    1. **Uploader** une image contenant une ou plusieurs poubelles
    2. **Cliquer** sur "Lancer l'Analyse IA" 
    3. **Visualiser** les résultats de détection et classification
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# Footer amélioré
# ---------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Détection Intelligente de Poubelles</strong> · Propulsé par YOLOv8 · Classification Vide/Pleine</p>
    <p style='font-size: 0.9rem;'>L'IA identifie automatiquement le statut de remplissage des poubelles</p>
</div>
""", unsafe_allow_html=True)