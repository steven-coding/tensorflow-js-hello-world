# Cat/Not-Cat Binary Image Classifier

Binary image classifier using a custom CNN built with TensorFlow.js (`@tensorflow/tfjs-node-gpu`).
Classifies whether an image contains a cat (1) or not (0).

## CNN Architecture

```
Input (128x128x3)
  → Conv2D (16 filters, 3x3, ReLU, L2=0.001) → MaxPooling2D (2x2) → Dropout (0.25)
  → Conv2D (32 filters, 3x3, ReLU, L2=0.001) → MaxPooling2D (2x2) → Dropout (0.25)
  → Flatten
  → Dropout (0.5)
  → Dense (64, ReLU, L2=0.001) → Dropout (0.3)
  → Dense (1, Sigmoid)                                        Output: probability [0, 1]
```

- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy
- **Regularization**: L2 (0.001) on all conv and dense layers
- **Dropout**: 0.25 after conv blocks, 0.5 and 0.3 in classification head

## Image Dimensions: 128x128 px (RGB)

- Smallest practical size for good CNN accuracy on binary classification
- 224x224 would be needed for transfer learning (MobileNet/ResNet) but is overkill here
- 3 color channels (RGB), normalized to [0, 1]

Use `src/util/resize_images.ts` to batch-resize raw images to 128x128 before training.

## Threshold: 0.5

- Sigmoid output >= 0.5 → cat (1)
- Sigmoid output < 0.5 → not cat (0)
- Standard best practice for balanced binary classification

## Training Data

### Positive images (cat/): ~1,000 images

**Source**: Microsoft Cats vs Dogs Dataset
- Download: https://www.microsoft.com/en-us/download/details.aspx?id=54765
- Hugging Face: `microsoft/cats_vs_dogs`

### Negative images (not_cat/): ~1,000 images — DIVERSE!

Only using dogs as negatives teaches "cat vs dog", not "cat vs everything".
Negatives should cover diverse categories:

| Category           | Share  | Count | Why                                    |
|--------------------|--------|-------|----------------------------------------|
| Dogs               | ~30%   | ~300  | Most common confusion object           |
| Other animals      | ~20%   | ~200  | Birds, fish, horses — fur/eyes similar |
| People/faces       | ~15%   | ~150  | Most common image category in practice |
| Objects/interiors   | ~15%   | ~150  | Cars, furniture, food                  |
| Landscapes/nature  | ~10%   | ~100  | Outdoor scenes without animals         |
| Abstract/textures  | ~10%   | ~100  | Patterns that may confuse edge detection |

**Sources**: Microsoft Cats vs Dogs (dogs), COCO Dataset, Open Images Dataset

### Data Split: Train / Validation / Test (70/15/15)

| Bucket          | Purpose                                         | Min images/class |
|-----------------|-------------------------------------------------|------------------|
| **train/**      | Model learns from this data                     | 700              |
| **validation/** | Monitors overfitting during training (val_loss)  | 150              |
| **test/**       | Final evaluation on never-seen data             | 150              |

### Expected directory structure

```
data/is_cat/
  train/
    cat/          ← cat images (.jpg, .jpeg, .png)
    not_cat/      ← diverse non-cat images
  validation/
    cat/
    not_cat/
  test/
    cat/
    not_cat/
```

## Training Parameters

| Parameter        | Value | Notes                              |
|------------------|-------|------------------------------------|
| Epochs           | 50    | With early stopping (patience 5)   |
| Batch Size       | 32    | Standard for GPU training          |
| Max Images/Class | 1000  | Limits GPU memory usage            |

## Execution Flow

1. **Train**: Loads train + validation splits, trains CNN, saves model
2. **Evaluate**: Runs `model.evaluate()` on test split, prints loss, accuracy, confusion matrix
3. **Predict**: Demo prediction on a single test image

## Model Persistence

Trained models are saved to `models/is_cat/` (model.json + weights.bin).
On subsequent runs, the saved model is loaded instead of retraining.

## Usage

1. Place images in `data/is_cat/{train,validation,test}/{cat,not_cat}/`
2. Run `npm start` — trains the model, evaluates on test data, runs demo prediction
3. On next run, the saved model is loaded automatically (evaluation + prediction still run)
