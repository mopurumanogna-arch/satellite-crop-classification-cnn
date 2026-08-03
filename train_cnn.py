"""
train_cnn.py

Trains a Convolutional Neural Network (CNN) - a type of deep
learning model built specifically for images - to classify real
Sentinel-2 satellite images into 10 land-cover classes, including
Annual Crop and Permanent Crop.

Run download_data.py and explore_data.py first, then run this file.

NOTE ON TIME: training on a normal laptop CPU (no GPU) with all
~27,000 images will take a while - roughly 15-40 minutes for the
10 epochs set below, depending on your computer. If you want a
faster first run to make sure everything works, lower EPOCHS to
2 or 3 first, confirm it runs end to end, then increase it and
run again for your real result.

Saves:
  training_curves.png       - accuracy/loss over each epoch
  confusion_matrix.png      - where the model got confused
  eurosat_cnn_model.keras   - the trained model itself
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = os.path.join("eurosat_data", "2750")
IMG_SIZE = (64, 64)   # EuroSAT images are already 64x64 pixels
BATCH_SIZE = 32
EPOCHS = 10            # lower this for a faster first test run

if not os.path.isdir(DATA_DIR):
    print(f"Could not find {DATA_DIR}. Run download_data.py first.")
    raise SystemExit

# ---- Step 1: load images straight from the folder structure ----
# Keras can automatically turn "one folder per class" into a
# labeled dataset - we don't have to manually build a table like
# we did with the earlier projects.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"\nClasses found: {class_names}\n")

# cache and prefetch - this just makes training faster by preparing
# the next batch of images while the current one is still training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---- Step 2: build the CNN ----
# Each Conv2D layer learns to detect visual patterns (edges,
# textures, shapes). Each MaxPooling2D layer shrinks the image
# while keeping the important information, which both speeds up
# training and helps the model generalize.
model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.3),  # randomly turns off 30% of connections during training to reduce overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---- Step 3: train ----
print(f"\nTraining for {EPOCHS} epochs. This will take a while - let it run.\n")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ---- Step 4: plot training curves ----
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy over Training")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss over Training")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()
print("Saved chart: training_curves.png")

# ---- Step 5: confusion matrix on the validation set ----
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=90)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Satellite Land Cover Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved chart: confusion_matrix.png")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ---- Step 6: save the trained model ----
model.save("eurosat_cnn_model.keras")
print("\nSaved trained model: eurosat_cnn_model.keras")
print("\nDone!")
