import pandas as pd
import numpy as np

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

SAMPLE_SIZE = 5000
SEED = 42


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading IMDB dataset...")

df = pd.read_csv(DATA_PATH)

print("Total reviews:", len(df))


# ==========================================
# 3. TAKE SAMPLE
# ==========================================

df = df.sample(
    n=SAMPLE_SIZE,
    random_state=SEED
).reset_index(drop=True)

print("Reviews used:", len(df))


# ==========================================
# 4. CONVERT LABELS
# ==========================================

df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

df = df[["review", "label"]]


# ==========================================
# 5. CREATE DATASET
# ==========================================

dataset = Dataset.from_pandas(df)


# ==========================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# First split:
# 80% training
# 20% temporary

split_1 = dataset.train_test_split(
    test_size=0.20,
    seed=SEED
)

train_dataset = split_1["train"]
temp_dataset = split_1["test"]


# Split temporary 50/50:
# 10% validation
# 10% final test

split_2 = temp_dataset.train_test_split(
    test_size=0.50,
    seed=SEED
)

validation_dataset = split_2["train"]
test_dataset = split_2["test"]


print("\nDataset split:")
print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Final Test:", len(test_dataset))


# ==========================================
# 7. LOAD TOKENIZER
# ==========================================

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 8. TOKENIZATION
# ==========================================

def tokenize_function(examples):

    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove original text column

train_dataset = train_dataset.remove_columns(
    ["review"]
)

validation_dataset = validation_dataset.remove_columns(
    ["review"]
)

test_dataset = test_dataset.remove_columns(
    ["review"]
)


# ==========================================
# 9. LOAD DISTILBERT
# ==========================================

print("\nLoading DistilBERT model...")

id2label = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

label2id = {
    "NEGATIVE": 0,
    "POSITIVE": 1
}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)


# ==========================================
# 10. METRICS
# ==========================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================
# 11. TRAINING SETTINGS
# ==========================================

training_args = TrainingArguments(

    output_dir="./distilbert_results",

    # Evaluate after every epoch
    eval_strategy="epoch",

    # IMPORTANT:
    # Do not save checkpoints during training
    save_strategy="no",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=3,

    weight_decay=0.01,

    logging_steps=50,

    report_to="none",

    use_cpu=True,

    seed=SEED
)


# ==========================================
# 12. TRAINER
# ==========================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics
)


# ==========================================
# 13. TRAIN
# ==========================================

print("\n" + "=" * 60)
print("STARTING FINAL DISTILBERT TRAINING")
print("=" * 60)

trainer.train()


# ==========================================
# 14. VALIDATION RESULT
# ==========================================

print("\n" + "=" * 60)
print("FINAL VALIDATION EVALUATION")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=validation_dataset
)

print("\nVALIDATION RESULTS")

print("=" * 60)

print(
    f"Accuracy : "
    f"{validation_results['eval_accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{validation_results['eval_precision']:.4f}"
)

print(
    f"Recall   : "
    f"{validation_results['eval_recall']:.4f}"
)

print(
    f"F1 Score : "
    f"{validation_results['eval_f1']:.4f}"
)


# ==========================================
# 15. FINAL TEST
# ==========================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)


# ==========================================
# 16. DISPLAY FINAL TEST RESULTS
# ==========================================

print("\nFINAL DISTILBERT TEST RESULTS")

print("=" * 60)

print(
    f"Accuracy : "
    f"{test_results['eval_accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{test_results['eval_precision']:.4f}"
)

print(
    f"Recall   : "
    f"{test_results['eval_recall']:.4f}"
)

print(
    f"F1 Score : "
    f"{test_results['eval_f1']:.4f}"
)


# ==========================================
# 17. SAVE FINAL MODEL
# ==========================================

print("\n" + "=" * 60)
print("SAVING FINAL DISTILBERT MODEL")
print("=" * 60)

trainer.save_model(
    "./distilbert_model"
)

tokenizer.save_pretrained(
    "./distilbert_model"
)

print("\nFinal DistilBERT model saved!")

print("Location: distilbert_model")

print("\nTraining and evaluation completed! 🎉")