# 📰🏀⚖️ Sports vs. Politics News Classification

This repository contains the end-to-end implementation of a high-performance Natural Language Understanding (NLU) system designed to categorize news articles into **Sport** or **Politics** domains.

By evaluating six machine learning architectures across multiple feature representations, this project identifies the optimal pipeline for topical separability in digital journalism.

---

## 🚀 Peak Accuracy: **96.08%**
**Logistic Regression + TF-IDF**

---

## 📌 Project Overview

In an era of information overload, automated text categorization is essential for effective content discovery.

This project utilizes a **Master Dataset (N = 7,658)** synthesized from four primary streams — including real-time web scraping and benchmark newsgroups — to train robust classifiers.

---

## 🔍 Key Features

- **Multi-Channel Data Acquisition:** Integrated data from BBC/Guardian (scraped), 20 Newsgroups, HuffPost, and Kaggle repositories.
- **Advanced Preprocessing:** Implemented a sequential cleaning pipeline using NLTK, including normalization, stop-word filtration, and WordNet lemmatization.
- **Comparative Feature Engineering:** Systematically compared **Bag of Words (BoW)**, **N-Grams**, and **TF-IDF** vectorization.
- **Extensive Model Benchmarking:** Evaluated 18 distinct experimental configurations (6 models × 3 representations).

---

## 📂 Repository Structure

The project is organized into a modular pipeline to ensure reproducibility:

```bash
├── plots/                    # Performance heatmaps, ROC/PR curves, and confusion matrices
├── sports_politics_data/     # Raw and processed text corpus (scraped and benchmark data)
├── trained_models/           # Saved .pkl files for all 18 model/feature combinations
├── scrape_bbc.py             # Web crawler for real-time journalism extraction
├── kaggle_data.py            # Converter for JSON-based news datasets (HuffPost)
├── 20newsgroup_data.py       # Integration script for 20 Newsgroups benchmark
├── preprocessing.py          # NLTK-based cleaning and lemmatization pipeline
├── final_dataset.py          # Merges master data components into unified dataset (v2)
├── model_training.py         # Training script for 6 ML architectures
├── data_description.py       # Generates statistical plots and class analysis
└── model_comparison_plots.py # Generates performance heatmaps and robustness curves
```

---

## 📊 Performance Analysis

The system identifies **Logistic Regression with TF-IDF** as the superior architecture.

Notably, feature scaling proved critical for instance-based learners — the **KNN model saw a 20% accuracy surge** when transitioning from BoW to TF-IDF.

### 📈 Top Model Metrics (TF-IDF Representation)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.9608** | **0.9615** | **0.9608** | **0.9608** |
| Naive Bayes | 0.9589 | 0.9595 | 0.9589 | 0.9588 |
| SVM (Linear) | 0.9569 | 0.9570 | 0.9569 | 0.9569 |

---

## 📊 Visual Validation

- **ROC-AUC Curves:** Achieved scores **> 0.99** for top-tier models, indicating near-perfect topical separability.
- **Accuracy Heatmap:** Demonstrates the critical dependency of KNN on TF-IDF weighting.
- **Confusion Matrices:** Detailed error analysis highlighting baseline KNN-BoW misclassification patterns.

---

## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/news-classification-nlu.git
cd news-classification-nlu
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk beautifulsoup4 requests
```

### 3️⃣ Run the Pipeline

- **Step 1:** Scrape and prepare data using `scrape_bbc.py` and `kaggle_data.py`.
- **Step 2:** Generate the final corpus with `20newsgroup_data.py` and `final_dataset.py`.
- **Step 3:** Train models and evaluate using `model_training.py` and `model_comparison_plots.py`.

---

## 🎓 Academic Context

- **Developer:** Vandita Gupta (B23CM1061)
- **Institution:** Indian Institute of Technology Jodhpur
- **Course:** Natural Language Understanding (NLU)

---
