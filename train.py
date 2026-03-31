import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

# ─── CONFIG ─────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# ✅ FIXED PATH
DATA_DIR = "dataset_scripts/dataset/animals/animals"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "animal_cnn.keras")
NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# ─── LOAD DATA ──────────────────────────
def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # Normalize
    normalization = layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    return train_ds, val_ds, class_names, num_classes


# ─── MODEL ─────────────────────────────
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # freeze

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model


# ─── TRAIN ─────────────────────────────
def train():
    train_ds, val_ds, class_names, num_classes = load_data()

    print(f"✅ Classes: {num_classes}")

    model = build_model(num_classes)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    # Save class names
    with open(NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    print("✅ Training complete!")
    print(f"📁 Model saved: {MODEL_PATH}")


# ─── RUN ───────────────────────────────
if __name__ == "__main__":
    train()