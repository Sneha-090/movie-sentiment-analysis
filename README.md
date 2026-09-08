# 🎬 Movie Sentiment Analysis using DistilBERT

A movie sentiment analysis project that uses a fine-tuned **DistilBERT** transformer model to classify movie reviews as **Positive** or **Negative**.

The model was fine-tuned on the **IMDB Movie Review Dataset** and integrated into a **Streamlit web application** for real-time sentiment prediction.

---

## 🚀 Project Overview

The goal of this project is to build a sentiment analysis system that can understand the sentiment expressed in movie reviews.

The project was initially developed using a traditional **TF-IDF + Machine Learning** approach and was later improved by fine-tuning **DistilBERT**, a transformer-based language model.

The final DistilBERT model achieved **91.16% accuracy** and **91.29% F1 score** on the unseen IMDB test dataset.

---

## 📊 Dataset

This project uses the **IMDB Movie Reviews Dataset** containing 50,000 labelled movie reviews.

The labelled dataset was divided into training, validation, and final test sets:

| Dataset | Samples |
|---|---:|
| Training | 20,000 |
| Validation | 5,000 |
| Final Test | 25,000 |
| **Total** | **50,000** |

The original IMDB training set was split into training and validation data using an **80/20 split**.

The final test set was kept separate and was used only for evaluating the trained model.

---

## 🧠 Model

The final model is based on:

**DistilBERT — distilbert-base-uncased**

DistilBERT is a smaller and faster transformer model derived from BERT. It was fine-tuned specifically for binary movie sentiment classification.

### Training Configuration

- **Model:** DistilBERT
- **Base Model:** distilbert-base-uncased
- **Dataset:** IMDB Movie Reviews
- **Training Samples:** 20,000
- **Validation Samples:** 5,000
- **Test Samples:** 25,000
- **Epochs:** 3
- **Batch Size:** 16
- **Mixed Precision:** FP16
- **GPU:** NVIDIA Tesla T4
- **Task:** Binary Sentiment Classification

---

## 📈 Final Test Results

The final model was evaluated on the separate **25,000-review IMDB test set**.

| Metric | Score |
|---|---:|
| **Accuracy** | **91.16%** |
| **Precision** | **89.94%** |
| **Recall** | **92.69%** |
| **F1 Score** | **91.29%** |

### Result

The model correctly classified approximately **91 out of every 100 unseen movie reviews**.

---

## 🔄 Model Pipeline

```text
Movie Review
      ↓
DistilBERT Tokenizer
      ↓
Fine-Tuned DistilBERT
      ↓
Sentiment Classification
      ↓
Positive / Negative
      ↓
Confidence Score
```

---

## 💻 Streamlit Application

The trained DistilBERT model is integrated into a Streamlit web application.

Users can enter a movie review and receive:

- Predicted sentiment
- Confidence score

### Example

```text
Review:
This movie was absolutely amazing. I loved every moment of it!

Prediction:
POSITIVE

Confidence:
99.57%
```

Another example:

```text
Review:
This movie was boring and disappointing.

Prediction:
NEGATIVE

Confidence:
99.76%
```

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **DistilBERT**
- **Streamlit**
- **Scikit-learn**
- **Git**
- **GitHub**

---

## 📁 Project Structure

```text
SENTIMENT_project/
│
├── app.py
├── predict.py
├── evaluate_transformer.py
├── README.md
├── requirements.txt
│
├── data/
├── distilbert_model/
├── distilbert_results/
│
├── sentiment_model.pkl
└── tfidf_vectorizer.pkl
```

### Important

The `distilbert_model/` and `distilbert_results/` directories are excluded from GitHub because the trained model files are large.

The trained model is stored locally for running the application.

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sneha-090/movie-sentiment-analysis.git
```

### 2. Navigate to the project

```bash
cd movie-sentiment-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the prediction script

```bash
python predict.py
```

The script allows you to enter movie reviews directly in the terminal.

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

The application will open in your browser.

---

## 🔬 Model Development

The project followed two approaches:

### 1. Traditional Machine Learning

The initial version used:

```text
Movie Review
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Model
     ↓
Sentiment Prediction
```

### 2. Transformer-based Approach

The model was then improved using fine-tuned DistilBERT:

```text
Movie Review
     ↓
DistilBERT Tokenization
     ↓
Fine-Tuning
     ↓
Evaluation
     ↓
91.16% Test Accuracy
```

The transformer-based approach provided better performance and allowed the model to capture contextual information in movie reviews more effectively.

---

## 📌 Key Features

- Binary movie sentiment classification
- Fine-tuned DistilBERT model
- 91.16% test accuracy
- Confidence score for predictions
- Interactive Streamlit interface
- Terminal-based prediction script
- Separate training, validation, and test datasets

---

## 🔮 Future Improvements

- Deploy the Streamlit application online
- Add probability visualization
- Add support for longer reviews
- Experiment with other transformer models
- Improve handling of mixed or neutral sentiment
- Compare DistilBERT with other transformer architectures
- Add more detailed model evaluation and visualizations

---

## 👩‍💻 Author

**Sneha Dubey**

B.Tech CSE (AI/ML)

GitHub:  
https://github.com/Sneha-090

---

## ⭐ Acknowledgements

- [Hugging Face](https://huggingface.co/)
- [IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)