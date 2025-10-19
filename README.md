# Smart-Product-Pricing-
Machine Learning solution for Amazon ML Challenge 2025 — Predicting optimal product prices using text and image features.
---

## 🧠 Smart Product Pricing – Amazon ML Challenge 2025

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Competition-Amazon%20ML%20Challenge%202025-orange)
![Model](https://img.shields.io/badge/Model-LightGBM%20|%20Transformer%20|%20CNN-yellow)

---

### 📄 Overview

This repository contains **Team SRM’s official submission** for the **Amazon ML Challenge 2025**.
The challenge involves predicting the **optimal price of e-commerce products** based on both **textual product details** and **image features**.

Our solution integrates **Natural Language Processing (NLP)** and **Computer Vision (CV)** techniques into a unified machine learning pipeline to analyze product descriptions and images holistically for accurate price prediction.

---

### 👥 Team SRM

| Name                   | Role                              |
| ---------------------- | --------------------------------- |
| Maharshith Narsapuram  | **Team Lead / Model Development** |
| Sibani Tiwari          | Data Preprocessing & Validation   |
| S Ruthveej Rao         | Model Optimization & Training     |
| Nampalli Eswara Prasad | Documentation & Testing           |

---

### 🎯 Problem Statement

E-commerce pricing involves multiple factors — brand, quantity, specifications, and visual appeal.
The task is to build an ML model that **predicts the product price** based on:

* **`catalog_content`** → Product title, description, and item pack quantity (text field).
* **`image_link`** → Product image URL (used for visual feature extraction).
* **`price`** → Target variable (only in training data).

---

### 📦 Dataset

| File                  | Description                          |
| --------------------- | ------------------------------------ |
| `train.csv`           | Training data (with `price`)         |
| `test.csv`            | Test data (no `price`)               |
| `sample_test_out.csv` | Sample submission format             |
| `src/utils.py`        | Provided utility for image downloads |

**Dataset size:**

* 75,000 train samples
* 75,000 test samples

---

### ⚙️ Approach Summary

| Component               | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| **Text Processing**     | TF-IDF + SVD and Sentence Transformer embeddings for semantic representation |
| **Image Features**      | EfficientNet-B0 / ResNet18 embeddings extracted via PyTorch                  |
| **Feature Engineering** | Item Pack Quantity (IPQ), text length, word count, brand name parsing        |
| **Modeling**            | LightGBM + Ensemble blending for robust predictions                          |
| **Optimization**        | Log-transform of price and SMAPE-based validation                            |
| **Evaluation Metric**   | SMAPE (Symmetric Mean Absolute Percentage Error)                             |

---

### 📈 SMAPE Formula

[
\text{SMAPE} = \frac{1}{N} \sum \frac{|y_{\text{pred}} - y_{\text{true}}|}{(|y_{\text{pred}}| + |y_{\text{true}}|)/2} \times 100
]

---

### 🧩 Repository Structure

```
📁 SmartProductPricing/
│
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_test_out.csv
│
├── src/
│   ├── utils.py                    # Helper for image download
│   ├── baseline_train_predict.py   # Main pipeline for training & prediction
│
├── check_dataset_correctness.py    # Dataset validation script
├── Documentation_AmazonMLChallenge_SRM.md  # 1-page methodology document
├── test_out.csv                    # Final submission file
└── README.md                       # Project overview (this file)
```

---

### 🚀 How to Run

#### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/SmartProductPricing.git
cd SmartProductPricing
```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Validate dataset

```bash
python check_dataset_correctness.py
```

#### 4️⃣ Train and generate predictions

```bash
python src/baseline_train_predict.py
```

#### 5️⃣ Output file

```
test_out.csv
```

This file matches the exact format of `sample_test_out.csv`:

| sample_id | price  |
| --------- | ------ |
| 10001     | 249.99 |
| 10002     | 189.00 |
| ...       | ...    |

---

### 📊 Model Performance

| Metric             | Score                                       |
| ------------------ | ------------------------------------------- |
| SMAPE (Validation) | **≈ 18.4%**                                 |
| Training Data      | 75,000 samples                              |
| Test Predictions   | 75,000 rows (fully aligned with `test.csv`) |

---

### 🧠 Technologies Used

* **Python 3.10+**
* **LightGBM**
* **scikit-learn**
* **PyTorch**
* **Sentence-Transformers**
* **NumPy / Pandas**
* **Matplotlib / tqdm**

---

### 🧾 License

This project is licensed under the **MIT License** in accordance with Amazon ML Challenge 2025 rules.

---

### 🙌 Acknowledgements

* **Amazon ML Challenge 2025** organizers
* **SRM Institute of Science and Technology**
* Open-source frameworks: *LightGBM, PyTorch, HuggingFace Transformers, scikit-learn*

---

### 💡 Keywords

`machine-learning` • `price-prediction` • `nlp` • `computer-vision` • `lightgbm` • `amazon-ml-challenge` • `multimodal-model`

---
