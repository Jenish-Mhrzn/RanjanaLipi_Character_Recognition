import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import os
from PIL import Image


# PATHS
BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANJANA_MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'ranjana_v1.keras')
BINARY_MODEL_PATH   = os.path.join(BASE_DIR, 'model', 'ranjana_binary_filter_v1.keras')
CSV_PATH            = os.path.join(BASE_DIR, 'utils', 'labels.csv')
CLASS_NAMES_PATH    = os.path.join(BASE_DIR, 'model', 'class_names.json')


# LOAD MODELS + CSV + CLASS NAMES 
print("Loading models...")
ranjana_model = tf.keras.models.load_model(RANJANA_MODEL_PATH)
binary_model  = tf.keras.models.load_model(BINARY_MODEL_PATH)
print(f"Ranjana model loaded — input: {ranjana_model.input_shape}")
print(f"Binary model loaded  — input: {binary_model.input_shape}")

labels_csv = pd.read_csv(CSV_PATH)
label_map  = labels_csv.set_index('class_id').to_dict('index')
print(f"CSV loaded — {len(labels_csv)} classes")

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
print(f"class_names loaded — {len(class_names)} classes")

missing = [c for c in class_names if c not in label_map]
if missing:
    print(f"WARNING: unmatched classes: {missing}")
else:
    print("All class_names matched in CSV")


### CONFIG
TARGET_SIZE        = 64
BINARY_THRESHOLD   = 0.85  # below this-> reject as not Ranjana
CLASSIFY_THRESHOLD = 0.70   # below this-> low confidence warning
TEMPERATURE        = 2.5    # higher = softer confidence scores


### QUALITY CHECK
def check_image_quality(pil_image):
    """
    Returns (is_good, reason)
    Accepts PIL Image.
    """
    img  = np.array(pil_image.convert('L'))
    h, w = img.shape

    # Check 1 — minimum size
    if h < 20 or w < 20:
        return False, f"Image too small ({w}x{h}) — minimum 20x20"

    # Check 2 — not blank/uniform
    if np.std(img) < 5:
        return False, "Image is blank"

    # Check 3 — has actual content
    _, simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    white = np.sum(simple > 127) / simple.size
    black = 1 - white

    if white < 0.02:
        return False, "Image too dark — no character visible"
    if black < 0.02:
        return False, "Image too bright — no character visible"

    return True, "OK"

## PREPROCESSING
def preprocess_image(pil_image):
    """
    Mirrors training preprocessing .
    """
    img = np.array(pil_image.convert('L'))

    # Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Otsu threshold + inversion
    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Auto background correction
    white_ratio = np.sum(img > 127) / img.size
    if white_ratio > 0.5:
        img = cv2.bitwise_not(img)

    # Aspect ratio resize with center padding
    h, w    = img.shape
    scale   = TARGET_SIZE / max(h, w)
    new_w   = int(w * scale)
    new_h   = int(h * scale)    
    resized = cv2.resize(img, (new_w, new_h))

    padded  = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    x_off   = (TARGET_SIZE - new_w) // 2
    y_off   = (TARGET_SIZE - new_h) // 2
    padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # Normalize
    final = padded.astype("float32") / 255.0
    final = np.expand_dims(final, axis=-1)  # (64, 64, 1)
    final = np.expand_dims(final, axis=0)   # (1, 64, 64, 1)
    return final

# CORE PREDICTION — FULL PIPELINE
def predict_ranjana(pil_image):
    """
    Accepts PIL Image.
    Returns prediction
    Two stage pipeline:
      Stage 1 — Binary filter  (Ranjana vs not_Ranjana)
      Stage 2 — 62-class model (which character)
    """
    try:
        # Quality Check 
        is_good, reason = check_image_quality(pil_image)
        if not is_good:
            return {"error": f"Image quality issue: {reason}"}

        # Preprocess 
        img_array = preprocess_image(pil_image)

        # Stage 1: Binary Filter 
        ranjana_score = float(binary_model.predict(img_array, verbose=0)[0][0])

        if ranjana_score < BINARY_THRESHOLD:
            return {
                "error"        : "Not a Ranjana character",
                "ranjana_score": round(ranjana_score * 100, 2)
            }

        # Stage 2: Temperature Scaled Classification
        probs_original = ranjana_model.predict(img_array, verbose=0)[0]

        # Reverse softmax → apply temperature scaling
        log_probs = np.log(probs_original + 1e-10)
        scaled    = log_probs / TEMPERATURE
        probs     = tf.nn.softmax(scaled).numpy()

        confidence    = float(np.max(probs))
        predicted_idx = int(np.argmax(probs))
        predicted     = class_names[predicted_idx]

        # CSV Lookup
        info     = label_map[predicted]
        phonetic = info['phonetic']
        roman    = info['roman']
        category = info['category']

        # Top 5
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5 = []
        for idx in top5_idx:
            folder   = class_names[idx]
            csv_info = label_map.get(folder, {
                'phonetic': folder,
                'roman'   : folder,
                'category': 'unknown'
            })
            top5.append({
                'character' : csv_info['roman'],
                'phonetic'  : csv_info['phonetic'],
                'confidence': round(float(probs[idx]) * 100, 2)
            })

        # Low Confidence Check
        if confidence < CLASSIFY_THRESHOLD:
            return {
                'warning'          : 'Low confidence — unclear character',
                'roman'            : roman,
                'phonetic'         : phonetic,
                'category'         : category,
                'confidence'       : round(confidence * 100, 2),
                'ranjana_score'    : round(ranjana_score * 100, 2),
                'top_5_predictions': top5
            }

        return {
            'roman'            : roman,
            'phonetic'         : phonetic,
            'category'         : category,
            'confidence'       : round(confidence * 100, 2),
            'ranjana_score'    : round(ranjana_score * 100, 2),
            'top_5_predictions': top5
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}