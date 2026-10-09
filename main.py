from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI application
app = FastAPI(
    title="Tanglish Sentiment Analysis API",
    description="API for analyzing Romanized Tamil + English mixed text.",
    version="1.0.0"
)

# XLM-RoBERTa handles phonetic Latin-script Tamil tokens
MODEL_NAME = "./fine_tuned_tanglish"

print("Loading Tanglish tokenizer and pre-trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set num_labels=2 to match your fine-tuned checkpoint output layer
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)
model.eval()

# Updated label mapping for 2 output classes
LABEL_MAPPING = {0: "Negative", 1: "Positive"}

class TextPayload(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Tanglish Sentiment Analyzer API is up and running!"}

@app.post("/predict")
def predict_sentiment(payload: TextPayload):
    # Tokenize Tanglish input text
    inputs = tokenizer(
        payload.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "input_text": payload.text,
        "sentiment": LABEL_MAPPING[predicted_class],
        "confidence_score": round(confidence, 4)
    }