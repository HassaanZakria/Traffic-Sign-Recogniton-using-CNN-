"""
Traffic Sign Recognition using CNN
Dataset: Real GTSRB (German Traffic Sign Recognition Benchmark)
Folder structure expected:
    gtsrb/
    ├── Train/
    │   ├── 0/   *.png or *.ppm images
    │   ├── 1/
    │   └── ... (43 folders)
    └── Test/
        └── *.png or *.ppm images  (optional)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import json

# ── Config ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 43
IMG_SIZE    = 32
BATCH_SIZE  = 64
EPOCHS      = 20
GTSRB_DIR   = "gtsrb"          # <-- folder containing Train/ and Test/
VAL_SPLIT   = 0.2              # 20% of Train used for validation

SIGN_NAMES = {
    0:"Speed limit 20",    1:"Speed limit 30",    2:"Speed limit 50",
    3:"Speed limit 60",    4:"Speed limit 70",    5:"Speed limit 80",
    6:"End speed 80",      7:"Speed limit 100",   8:"Speed limit 120",
    9:"No passing",       10:"No passing >3.5t", 11:"Right-of-way junction",
   12:"Priority road",   13:"Yield",             14:"Stop",
   15:"No vehicles",     16:"No vehicles >3.5t", 17:"No entry",
   18:"General caution", 19:"Danger curve left", 20:"Danger curve right",
   21:"Double curve",    22:"Bumpy road",        23:"Slippery road",
   24:"Road narrows R",  25:"Road work",         26:"Traffic signals",
   27:"Pedestrians",     28:"Children crossing", 29:"Bicycles crossing",
   30:"Ice/snow",        31:"Wild animals",      32:"End speed limits",
   33:"Turn right ahead",34:"Turn left ahead",   35:"Ahead only",
   36:"Ahead or right",  37:"Ahead or left",     38:"Keep right",
   39:"Keep left",       40:"Roundabout",        41:"End no passing",
   42:"End no passing >3.5t",
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_gtsrb_train(gtsrb_dir: str):
    """
    Load all images from gtsrb/Train/<class_id>/*.ppm (or .png).
    Returns numpy arrays X (N,32,32,3) float32 and y (N,) int.
    """
    train_dir = os.path.join(gtsrb_dir, "Train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Could not find '{train_dir}'.\n"
            f"Make sure your GTSRB folder is named 'gtsrb' and sits in the same "
            f"directory as train.py, with a 'Train' subfolder inside it."
        )

    images, labels = [], []
    classes_found = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))],
        key=lambda x: int(x)
    )
    print(f"  Found {len(classes_found)} class folders in {train_dir}")

    for cls_name in classes_found:
        cls_id  = int(cls_name)
        cls_dir = os.path.join(train_dir, cls_name)
        files   = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith((".ppm", ".png", ".jpg", ".jpeg"))]
        for fname in files:
            try:
                img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                images.append(np.array(img, dtype=np.uint8))
                labels.append(cls_id)
            except Exception as e:
                print(f"  Warning: skipping {fname} — {e}")

        if (cls_id + 1) % 10 == 0 or cls_id == len(classes_found) - 1:
            print(f"  Loaded class {cls_id:>2d}/{len(classes_found)-1}  "
                  f"({len(files)} images)")

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def load_gtsrb_test(gtsrb_dir: str):
    """
    Load test images from gtsrb/Test/*.ppm.
    Returns X array; labels are unknown unless a CSV is present.
    """
    test_dir = os.path.join(gtsrb_dir, "Test")
    if not os.path.isdir(test_dir):
        return None, None

    # Try to read labels from CSV if present
    csv_path = os.path.join(gtsrb_dir, "Test.csv")
    label_map = {}
    if os.path.isfile(csv_path):
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = os.path.basename(row.get("Path", row.get("Filename", "")))
                label_map[fname] = int(row.get("ClassId", -1))

    images, labels = [], []
    files = sorted([f for f in os.listdir(test_dir)
                    if f.lower().endswith((".ppm", ".png", ".jpg", ".jpeg"))])
    for fname in files:
        try:
            img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            images.append(np.array(img, dtype=np.float32) / 255.0)
            labels.append(label_map.get(fname, -1))
        except Exception as e:
            print(f"  Warning: skipping {fname} — {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


# ── Model ────────────────────────────────────────────────────────────────────

def build_model():
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # Block 1
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="TrafficSignCNN")
    return model


def augment_fn(x, y):
    x = tf.image.random_brightness(x, 0.2)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_flip_left_right(x)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


# ── Training ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Traffic Sign Recognition — CNN  (GTSRB Real Dataset)")
    print("=" * 60)

    # Load data
    print(f"\nLoading training data from '{GTSRB_DIR}/Train/' ...")
    X, y = load_gtsrb_train(GTSRB_DIR)
    print(f"  Total training images: {len(X)}")

    # Train / val split
    n_val   = int(len(X) * VAL_SPLIT)
    X_val,   y_val   = X[:n_val],  y[:n_val]
    X_train, y_train = X[n_val:],  y[n_val:]
    print(f"  Train: {len(X_train)}   Val: {len(X_val)}")

    # Optional test set
    print(f"\nLoading test data from '{GTSRB_DIR}/Test/' ...")
    X_test, y_test = load_gtsrb_test(GTSRB_DIR)
    if X_test is not None:
        has_test_labels = (y_test >= 0).all()
        print(f"  Test images: {len(X_test)}"
              f"{'  (with labels)' if has_test_labels else '  (no labels — skipping eval)'}")
    else:
        has_test_labels = False
        print("  No Test folder found — skipping test evaluation.")

    # Build model
    model = build_model()
    model.summary()
    print(f"\n  Trainable parameters: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # tf.data pipelines
    ds_train = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(4096)
        .batch(BATCH_SIZE)
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    os.makedirs("models", exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint("models/best_model.keras", save_best_only=True, verbose=0),
    ]

    print("\nTraining...\n")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Validation accuracy
    val_loss, val_acc = model.evaluate(ds_val, verbose=0)
    print(f"\nValidation — Loss: {val_loss:.4f}  Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")

    # Per-class accuracy on val
    preds = model.predict(X_val, verbose=0).argmax(axis=1)
    per_class = {}
    for cls in range(NUM_CLASSES):
        mask = y_val == cls
        if mask.sum() > 0:
            per_class[int(cls)] = round(float((preds[mask] == cls).mean()), 4)

    # Test evaluation (if labels available)
    test_acc_val = None
    if X_test is not None and has_test_labels:
        ds_test = (
            tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )
        t_loss, t_acc = model.evaluate(ds_test, verbose=0)
        test_acc_val = round(float(t_acc), 4)
        print(f"Test       — Loss: {t_loss:.4f}  Accuracy: {t_acc:.4f} ({t_acc*100:.1f}%)")

    # Save results
    results = {
        "val_accuracy":       round(float(val_acc), 4),
        "val_loss":           round(float(val_loss), 4),
        "test_accuracy":      test_acc_val,
        "epochs_trained":     len(history.history["accuracy"]),
        "train_acc":          [round(float(v), 4) for v in history.history["accuracy"]],
        "val_acc_history":    [round(float(v), 4) for v in history.history["val_accuracy"]],
        "train_loss_history": [round(float(v), 4) for v in history.history["loss"]],
        "val_loss_history":   [round(float(v), 4) for v in history.history["val_loss"]],
        "per_class_accuracy": per_class,
        "total_params":       model.count_params(),
        "dataset":            "GTSRB (real images)",
    }
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    model.save("models/traffic_sign_cnn.keras")
    print("\nModel  → models/traffic_sign_cnn.keras")
    print("Results→ models/results.json")
    print("Done!")


if __name__ == "__main__":
    train()
