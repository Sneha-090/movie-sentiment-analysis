import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. SETTINGS
# ==========================================

DATA_PATH = "data/IMDB Dataset.csv"

MODEL_NAME = "distilbert-base-uncased"

# CPU ke liye pehle small experiment
SAMPLE_SIZE = 2000


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading IMDB dataset...")

df = pd.read_csv(DATA_PATH)

print("Total reviews:", len(df))


# ==========================================
# 3. TAKE SMALL SAMPLE
# ==========================================

df = df.sample(
    n=SAMPLE_SIZE,
    random_state=42
).reset_index(drop=True)

print("Reviews used for DistilBERT:", len(df))


# Convert sentiment to numerical labels
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})


# Keep only required columns
df = df[["review", "label"]]


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

print("Training reviews:", len(train_dataset))
print("Testing reviews:", len(test_dataset))


# ==========================================
# 5. LOAD TOKENIZER
# ==========================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# tokenzation

def tokenize_function(examples):

    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


print("Tokenizing dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original text column
train_dataset = train_dataset.remove_columns(["review"])
test_dataset = test_dataset.remove_columns(["review"])


# load distilbert model

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# metrices

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions
    )

    recall = recall_score(
        labels,
        predictions
    )

    f1 = f1_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# training setting

training_args = TrainingArguments(

    output_dir="./distilbert_results",

    eval_strategy="epoch",

    save_strategy="no",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=50,

    report_to="none",

    use_cpu=True
)


# trainer

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# train

print("\n" + "=" * 60)
print("STARTING DISTILBERT TRAINING")
print("=" * 60)

trainer.train()



#  EVALUATE

print("\n" + "=" * 60)
print("EVALUATING DISTILBERT")
print("=" * 60)

results = trainer.evaluate()



#  DISPLAY RESULTS


print("\nDISTILBERT RESULTS")
print("=" * 60)

print(
    f"Accuracy : {results['eval_accuracy']:.4f}"
)

print(
    f"Precision: {results['eval_precision']:.4f}"
)

print(
    f"Recall   : {results['eval_recall']:.4f}"
)

print(
    f"F1 Score : {results['eval_f1']:.4f}"
)


#  SAVE MODEL


print("\nSaving DistilBERT model...")

trainer.save_model("./distilbert_model")
tokenizer.save_pretrained("./distilbert_model")

print("\nDistilBERT model saved!")
print("Location: distilbert_model")

print("\nTraining and evaluation completed! 🎉")