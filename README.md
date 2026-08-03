# satellite-crop-classification-cnn
# Satellite Land Cover & Crop Classification (CNN)

An image classification project that trains a Convolutional Neural
Network (CNN) to classify **real Sentinel-2 satellite images** into
10 land-cover types, including **Annual Crop** and **Permanent
Crop** - the same kind of task real remote sensing programs
(including ISRO's NRSC/FASAL crop monitoring work) are built around.

This is the deepest of your 4 projects: it works directly with
images (not tables of numbers) and uses deep learning, not
traditional ML models.


## The real-world problem
Sentinel-2 is a European Space Agency satellite that continuously
photographs the Earth's surface. Turning a raw satellite photo into
a useful label ("this is farmland," "this is a highway," "this is
forest") is a real, practical remote sensing task - used for crop
monitoring, urban planning, and land management. This project
trains a model to do exactly that, using the EuroSAT dataset: 27,000
real, labeled Sentinel-2 images across 10 classes.

## Pipeline overview
| Step | File | What it does |
|---|---|---|
| 1 | `download_data.py` | Downloads the real EuroSAT satellite image dataset (~90 MB) |
| 2 | `explore_data.py` | Shows class counts and sample images from each class |
| 3 | `train_cnn.py` | Builds and trains a CNN, evaluates it, saves the trained model |

## Setup and how to run
1. Install the required libraries:
   ```
   pip install requests tensorflow pillow scikit-learn matplotlib
   ```
   Note: `tensorflow` is a large install (a few hundred MB) and can
   take several minutes.
2. Run the three scripts in order:
   ```
   python download_data.py
   python explore_data.py
   python train_cnn.py
   ```
3. `download_data.py` needs internet access. Training in
   `train_cnn.py` does not, but it does take real time and CPU -
   don't close the terminal while it's running.

**Tip:** the first time you run `train_cnn.py`, consider opening
the file and changing `EPOCHS = 10` to `EPOCHS = 2` near the top,
just to confirm everything runs without errors quickly. Once
confirmed, change it back to 10 (or higher) and run it again for
your real result.

## Files generated when you run this
| File | What it shows |
|---|---|
| `eurosat_data/` | The downloaded satellite images, organized by class folder |
| `eda_class_distribution.png` | How many images exist per class |
| `eda_sample_images.png` | A real satellite image from each of the 10 classes |
| `training_curves.png` | How accuracy and loss changed over each training epoch |
| `confusion_matrix.png` | Which classes the model confuses with each other |
| `eurosat_cnn_model.keras` | The trained model file itself |




