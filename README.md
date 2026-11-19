PROJECT TITLE
An Adaptive Lightweight Deep Learning Network with Augmented Representation Learning for Offensive Text Detection

FRAMEWORK NAME
ILwYOLOv8Net (Multimodal Offensive Meme Detection)

TEAM AND INSTITUTION
- Program: M.C.A., School of Computing, SASTRA Deemed University
- Authors: Manisha A (127176064), Muthukumar K (127176070)
- Project Coordinator: Dr. A. Srilakshmi, Assistant Professor, SOC, SASTRA

BASE PAPER
- Title: Advanced success-driven lightweight deep learning network with data augmentation for detecting offensive text on unstructured social media data
- Journal: Knowledge-Based Systems (Scopus / SCI-E)

1. PROBLEM STATEMENT AND OBJECTIVES
Memes combine an image with a short caption, and the offensive meaning often appears only when both are read together. Text-only or image-only models miss this combined context, while heavy multimodal transformers (CLIP, VisualBERT, ViLBERT, OSCAR) require high computation.

Objectives:
- Build a lightweight multimodal framework to classify memes as Offensive (1) or Non-Offensive (0).
- Extract visual features using a YOLOv8 backbone and enhance multi-scale visual patterns using LPFP.
- Extract text features using TF-IDF followed by Truncated SVD dimensionality reduction.
- Fuse both modalities and classify with a Multi-Layer Perceptron (MLP).
- Address class imbalance cleanly and tune hyperparameters with AdSOA.

2. DATASET DETAILS
- Dataset: Facebook Hateful Memes Dataset (Meta / Facebook AI, via Kaggle)
- Raw dataset size: 8,500 memes (5,450 non-offensive, 3,050 offensive)
- After removing duplicate captions and missing images: 7,072 unique memes
- Training split (stratified 80%): 5,657 memes (3,765 non-offensive, 1,892 offensive)
- Validation split (stratified 20%): 1,415 memes (942 non-offensive, 473 offensive)
- Note: Dataset files are excluded from GitHub due to Meta's research-only license.

3. MODEL ARCHITECTURE (HOW IT WORKS)
A. Image Branch (768-D Visual Feature):
   - Input meme image is resized to 224x224 and normalized.
   - Pretrained YOLOv8n-cls backbone and neck (classification head removed, fine-tuned) extracts spatial features, followed by a 1x1 convolution producing a 1280 x 7 x 7 feature map.
   - LPFP (Lightweight Pyramid Feed Perceptron) applies a 5-level Laplacian pyramid, dilated convolutions (dilation 2, 4, 8), recursive depthwise-separable blocks, and channel + spatial dual attention, pooling down to a 768-dimensional visual vector.

B. Text Branch (512-D Text Feature):
   - Caption text is cleaned and vectorized with TF-IDF (max 20,000 terms, English stop words removed), fitted strictly on the training split to prevent data leakage.
   - Truncated SVD compresses the sparse TF-IDF vectors into a 512-dimensional dense text vector.

C. Fusion and Classifier:
   - Concatenation joins the 768-D visual vector and 512-D text vector into a 1280-D fused representation.
   - MLP Classifier (1280 -> 512 -> 1024 -> 2 with ReLU and 0.3 Dropout) outputs probabilities for Non-Offensive (0) and Offensive (1).
   - Total trainable parameters: ~100.8 Million.

4. METHODOLOGY AND KEY DECISIONS
- Class Imbalance (Class Weights vs. SMOTE): The notebook includes a Conditional-SMOTE routine on the 512-D minority text features, but in the final run SMOTE is disabled because pairing synthetic text vectors with reused images breaks true image-text alignment. Instead, Class-Weighted Cross-Entropy Loss is used (0.751 for non-offensive, 1.495 for offensive).
- Hyperparameter Search (AdSOA): Adaptive Search Optimization Algorithm (population 4, 2 iterations) tuned learning rate, dropout, and hidden size, selecting lr = 1e-4, dropout = 0.3, and hidden_dim = 512.
- Training Setup: Adam optimizer (weight decay 1e-4), backbone learning rate set to 0.1x base lr to preserve pretrained weights, batch size 16, max 8 epochs with early stopping (patience 3), mixed precision (AMP) on Kaggle Tesla T4 GPU, seed 42.

5. FINAL RESULTS
A. Validation Split (1,415 memes):
   - Accuracy: 69.33% (beats majority-class baseline of 66.57%)
   - Precision (Offensive): 53.91%
   - Recall (Offensive): 56.87%
   - F1-Score (Offensive): 55.35%
   - Specificity: 75.58%
   - MCC: 0.3204
   - AUROC: 0.7225
   - Average Precision: 0.5807
   - Log-Loss: 0.6052
   - Confusion Matrix: 712 True Negatives, 230 False Positives, 204 False Negatives, 269 True Positives.

B. 5-Fold Stratified Cross-Validation (Mean +/- Std):
   - Accuracy: 70.16% +/- 1.96%
   - Precision: 55.00% +/- 2.80%
   - Recall: 60.59% +/- 4.64%
   - F1-Score: 57.56% +/- 2.71%
   - Specificity: 74.97% +/- 3.44%
   - MCC: 0.3484 +/- 0.0386
   - AUROC: 0.7363 +/- 0.0228

6. LIMITATIONS AND FUTURE SCOPE
- TF-IDF + SVD relies on word frequencies and cannot deeply understand sarcasm, irony, or hidden cultural context.
- Future improvements: replace TF-IDF + SVD with contextual transformers (RoBERTa / DeBERTa), run ablation studies across modalities, and add Grad-CAM visual heatmaps for explainability.