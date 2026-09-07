from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "distilbert_model"

print("Loading trained DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully!\n")


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][prediction].item()

    # Check your model's actual label mapping
    label = model.config.id2label.get(prediction, str(prediction))

    return label, confidence


while True:
    text = input("Enter a sentence (or type 'exit'): ")

    if text.lower() == "exit":
        break

    label, confidence = predict_sentiment(text)

    print(f"\nPrediction : {label}")
    print(f"Confidence: {confidence * 100:.2f}%\n")