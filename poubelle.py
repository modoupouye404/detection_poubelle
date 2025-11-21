# app.py (partie corrigée)
import torch
from ultralytics import YOLO
import streamlit as st
import os

# ---------------------------------------
# 🧠 CHARGEMENT DU MODEL YOLO (VERSION CORRIGÉE)
# ---------------------------------------
MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.warning(f"📁 Modèle non trouvé à l'emplacement: {path}")
        return None
    try:
        # Solution pour PyTorch 2.6+ - Ajout des classes sûres
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import C2f, Bottleneck
        from ultralytics.nn.modules.head import Detect
        
        # Ajouter les classes Ultralytics aux globals autorisés
        torch.serialization.add_safe_globals([Conv, C2f, Bottleneck, Detect])
        
        # Charger le modèle
        model = YOLO(path)
        st.success(f"✅ Modèle chargé avec succès: {path}")
        return model
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        
        # Tentative de rechargement avec méthode alternative
        try:
            st.info("🔄 Tentative de chargement alternatif...")
            # Méthode directe avec torch.load en mode sécurisé
            weights = torch.load(path, weights_only=False)
            st.success("✅ Modèle chargé avec méthode alternative!")
            
            # Recréer le modèle YOLO avec les poids
            model = YOLO('yolov8n.pt')  # Modèle de base
            model.model.load_state_dict(weights)
            return model
            
        except Exception as e2:
            st.error(f"❌ Échec du chargement alternatif: {str(e2)}")
            return None

model = load_model()
