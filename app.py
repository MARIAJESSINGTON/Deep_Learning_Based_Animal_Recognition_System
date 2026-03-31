import json
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

IMG_SIZE = 224
MODEL_PATH = "model/animal_cnn.keras"
NAMES_PATH = "model/class_names.json"

# ─── Load Model ───────────────────────
print("⏳ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(NAMES_PATH) as f:
    class_names = json.load(f)

print("✅ Model loaded")

# ─── Emoji Map ───────────────────────
EMOJI = {
    "cat": "🐱", "dog": "🐶", "lion": "🦁", "tiger": "🐯",
    "elephant": "🐘", "horse": "🐴", "bear": "🐻",
    "default": "🐾"
}

def get_emoji(label):
    for key in EMOJI:
        if key in label.lower():
            return EMOJI[key]
    return EMOJI["default"]

# ─── Preprocess ──────────────────────
def prepare(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ─── Predict Function (TOP 5) ─────────
def predict(image):
    arr = prepare(image)
    preds = model.predict(arr)[0]

    # Top 5 predictions
    top_indices = np.argsort(preds)[::-1][:5]

    result = "🐾 **Prediction Results:**\n\n"

    for i, idx in enumerate(top_indices):
        label = class_names[idx]
        confidence = preds[idx] * 100
        emoji = get_emoji(label)

        if i == 0:
            result += f"👉 **{emoji} {label.title()} ({confidence:.2f}%)**\n\n"
        else:
            result += f"{emoji} {label.title()} ({confidence:.2f}%)\n"

    return result

# ─── Gradio UI ──────────────────────
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="markdown",
    title="🐾 Animal Classifier",
    description="Upload an image → AI will tell what animal it is"
)

# ─── Run ────────────────────────────
app.launch()