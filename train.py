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

print("Downloading Tanglish Sentiment Dataset...")
raw_dataset = load_dataset("community-datasets/tamilmixsentiment")

# Dataset Class Definition: 0 -> Positive, 1 -> Negative
label2id = {"Positive": 0, "Negative": 1}
id2label = {0: "Positive", 1: "Negative"}

# Filter dataset for Positive (0) and Negative (1)
filtered_train = raw_dataset["train"].filter(lambda x: x["label"] in [0, 1])
filtered_val = raw_dataset["validation"].filter(
    lambda x: x["label"] in [0, 1]
)

MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


tokenized_train = filtered_train.map(preprocess_function, batched=True)
tokenized_val = filtered_val.map(preprocess_function, batched=True)

# Load model configured with exact label maps
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


training_args = TrainingArguments(
    output_dir="./fine_tuned_tanglish",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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

print("Training model...")
trainer.train()

print("Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_tanglish")
tokenizer.save_pretrained("./fine_tuned_tanglish")
print("Saved successfully!")