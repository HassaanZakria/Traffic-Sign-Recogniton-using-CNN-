# Traffic Sign Recognition — CNN (GTSRB)

A complete deep learning project for recognizing **43 classes** of traffic signs using a CNN trained on the **real GTSRB dataset**.

## Project Structure

```
traffic_sign_cnn/
├── train.py           # Training pipeline — loads real GTSRB images
├── predict.py         # Inference script for any image
├── gtsrb/
│   ├── Train/
│   │   ├── 0/        ← .ppm images for class 0
│   │   ├── 1/
│   │   └── ...       ← 43 class folders total
│   └── Test/         ← optional test images
└── models/
    ├── traffic_sign_cnn.keras
    ├── best_model.keras
    └── results.json
```

## Quick Start

```bash
# 1. Train on GTSRB (make sure gtsrb/ folder is in same directory)
python train.py

# 2. Predict on any traffic sign image
python predict.py path\to\sign.jpg

# 3. Use a custom model path
python predict.py path\to\sign.jpg --model models\traffic_sign_cnn.keras
```

## CNN Architecture

```
Input (32×32×3)
    │
    ├── Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    ├── Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ├── Block 3: Conv2D(128) → BN → MaxPool → Dropout(0.25)
    ├── Flatten → Dense(256) → BN → Dropout(0.5)
    └── Dense(43, softmax)
```

**Parameters:** 676,427

## Key Techniques

- **Real GTSRB images** — 50,000+ actual traffic sign photos
- **BatchNormalization** — stable training, higher learning rates
- **Dropout** (0.25 conv / 0.5 FC) — prevents overfitting
- **Data augmentation** — brightness, contrast, horizontal flip
- **Adam + ReduceLROnPlateau** — adaptive learning rate
- **EarlyStopping** — auto-stops when val accuracy plateaus

## Expected Results on GTSRB

| Metric | Expected |
|--------|----------|
| Validation accuracy | ~97–99% |
| Test accuracy | ~96–98% |
