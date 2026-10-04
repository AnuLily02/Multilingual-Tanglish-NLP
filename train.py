import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the benchmark Tanglish dataset
print("Loading DravidianCodeMix Tanglish dataset...")
dataset = load_dataset("DravidianCodeMix", "tamil")

# Map text labels to integers: Negative (0), Neutral (1), Positive (2)
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# 2. Tokenization helper
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

# Filter dataset to standard 3 classes and encode labels
dataset = dataset.filter(lambda x: x["label"] in label2id)
dataset = dataset.map(lambda x: {"label_id": label2id[x["label"]]})
tokenized_ds = dataset.map(preprocess_function, batched=True)

# 3. Load model with class mappings
model = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base", 
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./tanglish_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
)

# 5. Fine-tune the model
print("Training model on Tanglish dataset...")
trainer.train()

# Save fine-tuned model and tokenizer locally
model.save_pretrained("./fine_tuned_tanglish")
tokenizer.save_pretrained("./fine_tuned_tanglish")
print("Model fine-tuned and saved successfully to './fine_tuned_tanglish'!")
