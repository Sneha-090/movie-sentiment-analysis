import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# -----------------------------
# Load trained DistilBERT model
# -----------------------------

MODEL_PATH = "Sneha-090/movie-sentiment-distilbert"

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------
# Prediction function
# -----------------------------

def predict_sentiment(review):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[0][prediction].item()

    label_map = {
        0: "NEGATIVE",
        1: "POSITIVE"
    }

    return label_map[prediction], confidence


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🎬 Movie Sentiment Analysis")

st.write(
    "Enter a movie review and the AI model will predict its sentiment."
)

review = st.text_area(
    "Enter your review"
)


if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        st.write(f"**Sentiment:** {sentiment}")

        st.write(
            f"**Confidence:** {confidence * 100:.2f}%"
        )