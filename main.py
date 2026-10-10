from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI(
    title="Tanglish Sentiment Analysis API",
    description="API for analyzing Romanized Tamil + English mixed text.",
    version="1.0.0",
)

MODEL_NAME = "./fine_tuned_tanglish"

print("Loading fine-tuned Tanglish model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.eval()

# Correct dataset mapping: Index 0 is Positive, Index 1 is Negative
LABEL_MAPPING = {0: "Positive", 1: "Negative"}


class TextPayload(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Tanglish Sentiment Analyzer API is up and running!"}


@app.post("/predict")
def predict_sentiment(payload: TextPayload):
    inputs = tokenizer(
        payload.text, return_tensors="pt", truncation=True, max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "input_text": payload.text,
        "sentiment": LABEL_MAPPING[predicted_class],
        "confidence_score": round(confidence, 4),
    }