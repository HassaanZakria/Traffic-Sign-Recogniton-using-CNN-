"""
Traffic Sign Recognition — Inference Script
Usage:
    python predict.py path\to\image.jpg
    python predict.py path\to\image.jpg --model models\traffic_sign_cnn.keras
"""
import os, sys, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 32

SIGN_NAMES = {
    0: "Speed limit 20 km/h",    1: "Speed limit 30 km/h",    2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h",    4: "Speed limit 70 km/h",    5: "Speed limit 80 km/h",
    6: "End of speed limit 80",  7: "Speed limit 100 km/h",   8: "Speed limit 120 km/h",
    9: "No passing",            10: "No passing >3.5t",       11: "Right-of-way at junction",
   12: "Priority road",        13: "Yield",                  14: "Stop",
   15: "No vehicles",          16: "No vehicles >3.5t",      17: "No entry",
   18: "General caution",      19: "Dangerous curve left",   20: "Dangerous curve right",
   21: "Double curve",         22: "Bumpy road",             23: "Slippery road",
   24: "Road narrows on right",25: "Road work",              26: "Traffic signals",
   27: "Pedestrians",          28: "Children crossing",      29: "Bicycles crossing",
   30: "Beware of ice/snow",   31: "Wild animals crossing",  32: "End all speed limits",
   33: "Turn right ahead",     34: "Turn left ahead",        35: "Ahead only",
   36: "Go straight or right", 37: "Go straight or left",   38: "Keep right",
   39: "Keep left",            40: "Roundabout mandatory",   41: "End of no passing",
   42: "End of no passing >3.5t",
}


def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def predict(image_path: str, model_path: str = r"models\traffic_sign_cnn.keras") -> dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Run train.py first to train and save the model."
        )

    model = tf.keras.models.load_model(model_path)
    x     = preprocess(image_path)
    probs = model.predict(x, verbose=0)[0]

    top5_idx = probs.argsort()[::-1][:5]
    return {
        "predicted_class": int(top5_idx[0]),
        "predicted_name":  SIGN_NAMES[top5_idx[0]],
        "confidence":      float(probs[top5_idx[0]]),
        "top5": [
            {"class": int(i), "name": SIGN_NAMES[i], "prob": float(probs[i])}
            for i in top5_idx
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition — Predict")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default=r"models\traffic_sign_cnn.keras",
                        help="Path to saved .keras model (default: models\\traffic_sign_cnn.keras)")
    args = parser.parse_args()

    result = predict(args.image, args.model)

    print(f"\n{'='*52}")
    print(f"  Traffic Sign Recognition — Result")
    print(f"{'='*52}")
    print(f"  Image      : {args.image}")
    print(f"  Prediction : [{result['predicted_class']:>2d}] {result['predicted_name']}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"\n  Top-5 Predictions:")
    for i, r in enumerate(result["top5"]):
        bar = "█" * int(r["prob"] * 25)
        print(f"  {i+1}. [{r['class']:>2d}] {r['name']:<35s}  {r['prob']*100:5.1f}%  {bar}")
    print(f"{'='*52}\n")
