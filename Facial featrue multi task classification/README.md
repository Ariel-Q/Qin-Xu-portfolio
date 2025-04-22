# ECSE552 Final Project — Multi-Label Facial Attribute Classification

This project investigates multi-label facial attribute classification using deep neural networks. We implement and compare two architectures:

- 🟨 A **CNN-based model** using a pretrained ResNet-50 backbone and a lightweight MLP head.
- 🟦 A **Transformer-based model** using EfficientFormerV2 with a region-wise grouped classification head.

Both models are trained and evaluated on the CelebA dataset. Metrics include accuracy, precision, recall, and macro-F1 score. We also analyze label noise and attribute difficulty by comparing performance across objective and subjective attribute subsets.

---

## 📁 Project Structure

```
ECSE552_FINAL_PROJECT/
├── ResNet-50/                        # CNN-based model
│   ├── train.py
│   ├── main_hengyi.ipynb
│   ├── models/
│   │   └── cnn_multilabel_model.py
│   ├── datasets/
│   │   └── celeb_A_dataset.py
│   ├── utils/
│   │   ├── preprocess.py
│   │   └── eval_utils.py
│   ├── data/
│   │   ├── list_attr_celeba.txt
│   │   ├── img_align_celeba/
│   │   └── train.csv / val.csv / celeba_multilabel.csv
│   └── checkpoints/

├── Transformer/                          # Transformer-based model
│   ├── main_Modi.ipynb
│   ├── mymodel.py
│   ├── efficientformer_v2.py
│   └── data/
│       └── trained_tf.pth

├── Plots/                         

├── Baseline/                        
│   ├── Baseline CNN.ipynb
│   ├── Baseline FCNN.ipynb

├── Baseline/                          
│   ├── Baseline CNN.ipynb
│   ├── Baseline FCNN.ipynb


├── requirements.txt
└── Evaluation and Comparison.ipynb
└── report.pdf
└── Plots
└── Results
```

---

## 🧠 Model Overview

### 🟨 CNN-Based Model (ResNet-50)

- ResNet-50 pretrained on ImageNet
- FC layer replaced with `nn.Identity()`
- Classification head:
  ```python
  nn.Sequential(
      nn.Linear(2048, 512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, 40)
  )
  ```

### 🟦 Transformer-Based Model (EfficientFormerV2)

- EfficientFormerV2-S2 variant from `timm`
- Last-stage feature map fed to grouped classification head
- Attributes are grouped by region (e.g., hair, eyes, mouth)
- Output: 40 logits → `Sigmoid` activation

---

## 🧪 Training & Evaluation

- Dataset: CelebA, using standard train / val / test split
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Batch Size: 32
- Evaluation metrics:
  - Accuracy, Precision, Recall
  - Macro-F1 (main metric)

Both models apply data augmentations:
```python
transforms.Compose([
    transforms.CenterCrop(178),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
```

---

## 🚀 How to Run

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare CelebA dataset:
Download the following from [CelebA official site](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):
- `img_align_celeba/`
- `list_attr_celeba.txt`

Then run preprocessing:
```bash
python ResNet-50/utils/preprocess.py
```

### 3. Train CNN model:
```bash
cd ResNet-50/
python train.py
```

### 4. Run Transformer (EfficientFormer) model:
```bash
cd Transformer/
jupyter notebook main.ipynb
```

---

## 📊 Output & Evaluation

- Best checkpoints saved to `checkpoints/`
- Final evaluation results in `cnn_test_results.csv` / notebook cells
- Figures for attribute performance, confusion, and noise sensitivity in report and notebooks

---

## 👨‍🎓 Authors

- Ariel Xu
- Hengyi Yin  
- Modi Fan 
Course: ECSE552 — Deep Learning for Engineering  
McGill University — Winter 2025

---

## 📄 License

This repository is intended for academic use only.
