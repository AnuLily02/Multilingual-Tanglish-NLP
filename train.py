import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

print("Downloading Tanglish Sentiment Dataset automatically from Hugging Face...")

# 1. Load the official community-hosted Tanglish dataset (No local CSV required!)
raw_dataset = load_dataset("community-datasets/tamilmixsentiment")

# Map integer labels in dataset to standard classes
# Label IDs: 0 -> Positive, 1 -> Negative, 2 -> Mixed_feelings, 3 -> unknown_state, 4 -> not-Tamil
# We filter for 0 (Positive) and 1 (Negative) to keep sentiment classification clean
LABEL_MAPPING = {0: "Positive", 1: "Negative"}

# Filter dataset to keep only Positive (0) and Negative (1) samples
filtered_train = raw_dataset["train"].filter(lambda x: x["label"] in [0, 1])
filtered_val = raw_dataset["validation"].filter(
    lambda x: x["label"] in [0, 1]
)

print(
    f"Dataset loaded successfully! Train samples: {len(filtered_train)}, Validation samples: {len(filtered_val)}"
)

# 2. Load XLM-RoBERTa Tokenizer
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


# Tokenize train and validation sets
tokenized_train = filtered_train.map(preprocess_function, batched=True)
tokenized_val = filtered_val.map(preprocess_function, batched=True)

# 3. Load Base Model configured for 2 classes (Negative: 0, Positive: 1)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "Positive", 1: "Negative"},
    label2id={"Positive": 0, "Negative": 1},
)


# Compute evaluation metrics during training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# 4. Configure Training Parameters
training_args = TrainingArguments(
    output_dir="./fine_tuned_tanglish",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 5. Start Training Process
print("\nStarting model training on Tanglish dataset...")
trainer.train()

# 6. Save Fine-Tuned Model locally
print("\nSaving fine-tuned model...")
model.save_pretrained("./fine_tuned_tanglish")
tokenizer.save_pretrained("./fine_tuned_tanglish")
print("Model fine-tuned and saved successfully to './fine_tuned_tanglish'!")