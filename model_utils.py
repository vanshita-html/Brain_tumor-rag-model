import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# Class names corresponding to model output index order [0, 1, 2, 3]
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

DISPLAY_LABELS = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor Detected',
    'pituitary': 'Pituitary Tumor'
}

COLOR_BADGES = {
    'glioma': {'bg': '#FEE2E2', 'color': '#DC2626', 'icon': '⚠️', 'type': 'danger'},
    'meningioma': {'bg': '#FEF3C7', 'color': '#D97706', 'icon': '⚠️', 'type': 'warning'},
    'notumor': {'bg': '#D1FAE5', 'color': '#059669', 'icon': '✅', 'type': 'success'},
    'pituitary': {'bg': '#FEF3C7', 'color': '#D97706', 'icon': '⚠️', 'type': 'warning'}
}

CLASS_DESCRIPTIONS = {
    'glioma': (
        "Gliomas are primary brain tumors that originate in the glial supportive cells "
        "(astrocytes, oligodendrocytes, ependymal cells). They range from low-grade (Grades 1-2) "
        "to high-grade (Grades 3-4, such as glioblastoma). Common clinical management includes safe surgical "
        "resection, targeted radiation therapy, and chemotherapy (e.g. Temozolomide)."
    ),
    'meningioma': (
        "Meningiomas arise from the meninges, the protective membranes surrounding the brain "
        "and spinal cord. Most meningiomas (~80–85%) are benign, slow-growing WHO Grade 1 tumors. "
        "Treatment depends on tumor size, location, and symptoms, often involving watchful observation "
        "or surgical removal."
    ),
    'notumor': (
        "No evidence of brain tumor tissue detected in this MRI scan slice. "
        "Cerebral parenchyma, ventricles, and anatomical structures appear within normal "
        "radiological parameters without abnormal mass effect."
    ),
    'pituitary': (
        "Pituitary tumors develop in the pituitary gland at the base of the brain behind the nasal cavity. "
        "Most are benign adenomas that may impact hormone regulation or cause visual disturbances "
        "by pressing on the optic chiasm. Management includes medication, endoscopic surgical resection, "
        "or radiotherapy."
    )
}

@st.cache_resource
def load_tumor_model(model_path="final_model.h5"):
    """
    Load Keras CNN model from disk with Streamlit resource caching.
    Falls back to best_model.h5 if primary path is missing.
    """
    if not os.path.exists(model_path):
        if os.path.exists("best_model.h5"):
            model_path = "best_model.h5"
        else:
            raise FileNotFoundError("Neither 'final_model.h5' nor 'best_model.h5' was found.")
    
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def preprocess_image(image_input, target_size=(128, 128)):
    """
    Preprocess image exactly as done during model training:
    - Resize to 128x128
    - Convert PIL image to RGB
    - Convert to numpy array & rescale by 1/255.0
    - Expand dimensions to (1, 128, 128, 3)
    """
    if hasattr(image_input, 'seek'):
        try:
            image_input.seek(0)
        except Exception:
            pass

    if isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = Image.open(image_input)
    
    img = img.convert("RGB")
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = img_array / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    
    return img, img_tensor

def predict_tumor(model, img_tensor):
    """
    Run inference on preprocessed tensor and return structured prediction dict.
    """
    preds = model.predict(img_tensor, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    raw_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[0][pred_idx] * 100.0)
    
    probabilities = {}
    for idx, cname in enumerate(CLASS_NAMES):
        probabilities[cname] = {
            'label': DISPLAY_LABELS[cname],
            'prob': float(preds[0][idx] * 100.0)
        }
    
    return {
        'class_key': raw_class,
        'label': DISPLAY_LABELS[raw_class],
        'confidence': round(confidence, 2),
        'probabilities': probabilities,
        'badge': COLOR_BADGES[raw_class],
        'description': CLASS_DESCRIPTIONS[raw_class]
    }
