**Multi-class classification for lung X-ray images (RSNA)**
**STAT 362 — Fall 2025**
**Team: Muse Miao, Nikole Montero Cervantes, Emily Kessel, Jackie Holman**

**Problem & Datase**

Chest X-rays are a common, low-cost tool for diagnosing pneumonia and other lung diseases, but interpretation can be subjective and time-consuming. In this project, we build a deep learning pipeline to automatically classify chest X-rays into three clinically meaningful categories: Normal, Pneumonia (Lung Opacity), and Other pathology.

We use the RSNA Pneumonia Detection Challenge (Kaggle) dataset curated by the Radiological Society of North America. Images are provided as DICOM files (one per patientId). We use stage_2_detailed_class_info.csv and stage_2_train_labels.csv, merge on patientId, drop duplicates so each patient appears once, and create a 3-class target mapping: Normal→0, Lung Opacity→1, No Lung Opacity/Not Normal→2. The final dataset contains 26,684 unique images with class counts 0=8,851, 1=6,012, 2=11,821 (slightly imbalanced, with class 2 as the largest class).

**Models**

Baseline: from-scratch CNN implemented in PyTorch.

Main deep learning model: DenseNet121 (ImageNet-pretrained) with a modified 3-class classification head. We train with cross-entropy loss (using class weighting for imbalance), AdamW optimization, early stopping based on validation accuracy, and image augmentation (horizontal flips + affine transforms).

**Key Results**

Best 3-class DenseNet121 model (held-out test set):
Test Accuracy: 0.7330
Balanced Accuracy: 0.7364

Binary (Healthy vs Sick; Sick = classes 1 or 2) using the same model predictions:
Accuracy: 0.8853
Precision: 0.9575
Recall: 0.8669
F1: 0.9099

**How to Run**

Install dependencies:

pip install -r requirements.txt

Run a representative demo (notebook workflow):

jupyter notebook
