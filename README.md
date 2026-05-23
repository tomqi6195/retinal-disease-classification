# Retinal Disease Classifier

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![EfficientNet](https://img.shields.io/badge/EfficientNet--B3-Transfer%20Learning-green.svg)

This project classifies retinal fundus images into 4 categories of eye health: normal, diabetic retinopathy, cataracts, and glaucoma. It uses publicly available Kaggle datasets for training, and explores multiple architectures including a baseline support vector machine (SVM), and a fine-tuned EfficientNet-B3 CNN with transfer learning and Grad-CAM visualization.

---

## Data Augmentation
During training, the retinal images were augmented using the following techniques:
1. **Random Horizontal Flip:** Each image had a 50% chance of being horizontally flipped to improve generalization across different eye orientations.
2. **Color Jitter:** Brightness, contrast, saturation, and hue were each randomly perturbed by ±0.1 to make the model robust to variations in imaging conditions and equipment.
3. **Resize & Normalize:** All images were resized to 512×512 and normalized using ImageNet mean and standard deviation values to ensure compatibility with the pre-trained backbone.

---

## Model Architectures & Pipeline

To tackle the complexity of retinal disease classification, this project uses a transfer learning approach with a well-established image classification backbone:

1. **The Baseline (SVM + ResNet-18 Features):** A pre-trained ResNet-18 was used to extract 512-dimensional feature vectors from each retinal image, which were then fed into an SVM with an RBF kernel for classification.
2. **The Primary Model (EfficientNet-B3):** Fine-tuned on ImageNet weights with a custom classification head — a dropout layer (p=0.3) followed by a fully connected layer outputting logits for 4 classes. Trained using the Adam optimizer, CrossEntropy loss, and a cosine annealing learning rate scheduler.
3. **Grad-CAM Visualization:** A gradient-weighted class activation mapping layer was added to the final model to highlight which regions of each retinal image most influenced the prediction, providing clinical interpretability for optometrists.

---

## Final Results

| Model Setup | Test Accuracy | Notable Strengths |
| :--- | :--- | :--- |
| Baseline SVM | 78.6% | Diabetic Retinopathy |
| EfficientNet-B3 (No Freeze, No Scheduler) | 91.7% | Baseline deep learning performance |
| EfficientNet-B3 (No Freeze, Cosine Annealing) | **93.8%** | **Best in-distribution accuracy** |
| EfficientNet-B3 (5-Fold CV, Out-of-Distribution) | 68.8% | Generalizes to higher-quality unseen datasets |

> **Note:** The in-distribution test set is a held-out split from the same Kaggle dataset used for training (Doddi, 2022). The out-of-distribution results were evaluated on a completely separate Kaggle dataset (K-S-Sanjay-Nithish, 2021) to test real-world generalization.

---

## How to Reproduce & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/tomqi6195/retinal-disease-classification.git
cd retinal-disease-classification
```

### 2. Download the Datasets
Download the training dataset from Kaggle:
- **Main dataset (Doddi):** https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
- **Out-of-distribution test dataset:** https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images

Place the unzipped datasets in the `data/` folder, following the structure below:
```
data/
├── cataract/
├── diabetic_retinopathy/
├── glaucoma/
└── normal/
```

### 3. Install Dependencies
```bash
pip install torch torchvision efficientnet-pytorch matplotlib numpy scikit-learn pillow
```

### 4. Run the Notebook
For hyperparameter tuning and in-distribution testing, open the tuning.ipynb notebook and run all cells. The notebook handles data splitting, training, evaluation, and Grad-CAM visualization end-to-end.

For out-of-distribution testing, use the testing.ipynb notebook.
