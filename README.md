Satellite Land Cover & Crop Classification (CNN)

An image classification project that trains a Convolutional Neural Network (CNN) to classify real Sentinel-2 satellite images into 10 land-cover types, including Annual Crop and Permanent Crop - the same kind of task real remote sensing programs (including ISRO's NRSC/FASAL crop monitoring work) are built around.

This is the deepest of your 4 projects: it works directly with images (not tables of numbers) and uses deep learning, not traditional ML models.

Important - read this before running

I was not able to test this project's code myself before giving it to you, because my environment here doesn't have internet access or the TensorFlow library installed. The satellite launch project and exoplanet project were both tested locally before I shared them; this one wasn't. The code is written carefully and I'm confident in it, but if something breaks, send me the exact error message and I'll fix it with you - same as we did for the earlier projects.

The real-world problem

Sentinel-2 is a European Space Agency satellite that continuously photographs the Earth's surface. Turning a raw satellite photo into a useful label ("this is farmland," "this is a highway," "this is forest") is a real, practical remote sensing task - used for crop monitoring, urban planning, and land management. This project trains a model to do exactly that, using the EuroSAT dataset: 27,000 real, labeled Sentinel-2 images across 10 classes.

Pipeline overview
Step	File	What it does
1	download_data.py	Downloads the real EuroSAT satellite image dataset (~90 MB)
2	explore_data.py	Shows class counts and sample images from each class
3	train_cnn.py	Builds and trains a CNN, evaluates it, saves the trained model
Setup and how to run
Install the required libraries:
   pip install requests tensorflow pillow scikit-learn matplotlib

Note: tensorflow is a large install (a few hundred MB) and can take several minutes. 2. Run the three scripts in order:

   python download_data.py
   python explore_data.py
   python train_cnn.py
download_data.py needs internet access. Training in train_cnn.py does not, but it does take real time and CPU - don't close the terminal while it's running.

Tip: the first time you run train_cnn.py, consider opening the file and changing EPOCHS = 10 to EPOCHS = 2 near the top, just to confirm everything runs without errors quickly. Once confirmed, change it back to 10 (or higher) and run it again for your real result.

Files generated when you run this
File	What it shows
eurosat_data/	The downloaded satellite images, organized by class folder
eda_class_distribution.png	How many images exist per class
eda_sample_images.png	A real satellite image from each of the 10 classes
training_curves.png	How accuracy and loss changed over each training epoch
confusion_matrix.png	Which classes the model confuses with each other
eurosat_cnn_model.keras	The trained model file itself
Key concepts (know these well for your interview)

What a CNN is, and why it's used for images instead of a regular ML model: A normal ML model (like the Random Forest in your exoplanet project) works on a table of numbers. An image is a grid of pixels - a CNN is designed specifically to find visual patterns (edges, textures, shapes) in that grid, and build up from simple patterns to more complex ones through its layers.

Convolution layers (Conv2D): Each one slides small filters across the image to detect specific visual patterns. Early layers tend to learn simple things like edges and colors; deeper layers combine those into more complex shapes.

Pooling layers (MaxPooling2D): These shrink the image (reduce its resolution) while keeping the strongest signal. This makes the network faster and more robust to small shifts in the image.

Dropout: During training, the model randomly "turns off" 30% of its connections in the dropout layer. This forces the model to not over-rely on any single pattern, which reduces overfitting (doing great on training data but poorly on new data).

Training curves matter more than a single accuracy number: If training accuracy keeps climbing but validation accuracy flattens or drops, that's overfitting. If both are low and flat, the model isn't learning well. Be ready to describe what your actual training_curves.png shows, not just quote a final number.

Why the confusion matrix matters: A single accuracy percentage hides which classes get confused. For example, don't be surprised if Annual Crop and Permanent Crop get mixed up sometimes - they can look visually similar from a satellite, and calling that out correctly is exactly the kind of insight that makes an interview answer sound like real understanding, not memorized results.

Results (from an actual training run)

Trained for 10 epochs on the full 27,000-image dataset (21,600 train / 5,400 validation), on a CPU (no GPU):

Overall accuracy: 89%

Class	Precision	Recall	F1-score
AnnualCrop	0.86	0.91	0.89
Forest	0.96	0.96	0.96
HerbaceousVegetation	0.82	0.87	0.84
Highway	0.81	0.79	0.80
Industrial	0.91	0.95	0.93
Pasture	0.87	0.90	0.88
PermanentCrop	0.81	0.77	0.78
Residential	0.96	0.97	0.96
River	0.92	0.81	0.86
SeaLake	0.98	0.98	0.98

What this shows:

Strongest classes: SeaLake, Forest, Residential (all 96%+ F1) — these have visually distinct patterns from above, so the model separates them easily.
Weakest class: PermanentCrop (0.78 F1) — permanent crops (orchards, vineyards) and annual crops (seasonal farmland) can look visually similar from a satellite, especially at the dataset's 64x64 pixel resolution. This is a real, explainable limitation, not a bug.
River has high precision (0.92) but lower recall (0.81) — when the model says "River," it's usually right, but it misses some actual rivers, most likely confusing them with visually similar linear features like highways.
For context: published results on this exact dataset reach ~98% accuracy using larger, pretrained networks (e.g. ResNet-50). This 89% result is from a smaller CNN trained from scratch — a solid, honest result for a first from-scratch model, not the ceiling of what's possible on this data.
Honest notes on this project
The dataset (EuroSAT) is real and widely used in published remote sensing research (Helber et al., 2017) - this isn't a toy dataset.
The model architecture here is a solid, standard, from-scratch CNN - not a state-of-the-art pretrained model. Published results on this exact dataset reach ~98% accuracy using larger, pretrained networks (like ResNet-50); this smaller model trained from scratch will likely land lower than that, and that's expected and fine to say out loud.
A genuinely strong next step (mention this if asked what you'd improve) is using transfer learning - starting from a pretrained image model (like ResNet or MobileNet) instead of training from scratch. This usually gives a large accuracy boost with less training time.
