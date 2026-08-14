from pathlib import Path
import re
from typing import List
import io
from io import BytesIO, StringIO

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "distilbert_sms_spam_model"

app = FastAPI(title="SMS Spam Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global tokenizer, model
    if not MODEL_DIR.exists():
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.to(device)
        model.eval()
        return True
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        return False

MODEL_LOADED = load_model()

class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    threshold: float = Field(0.5, ge=0.0, le=1.0)

def rule_scan(message: str) -> List[str]:
    text = message.lower()
    rules = []
    if re.search(r"https?://|www\.|bit\.ly|tinyurl|t\.co/", text):
        rules.append("Contains a URL or shortened link")
    if re.search(r"\b(free|win|won|winner|prize|reward|cash|claim|congratulations)\b", text):
        rules.append("Contains prize, reward, or winning language")
    if re.search(r"\b(urgent|act now|limited time|hurry|immediately|last chance)\b", text):
        rules.append("Uses urgency or pressure language")
    if re.search(r"\b(call|text|sms)\s+(now|today)|\bcall\s+\d{5,}\b", text):
        rules.append("Requests immediate contact")
    if re.search(r"[$£€]\s?\d+|\b\d+\s?(dollars|usd|rupees|inr)\b", text):
        rules.append("Contains money or monetary reward language")
    if re.search(r"\b(click|unsubscribe|opt out|reply)\b", text):
        rules.append("Contains action/response instructions")
    if re.search(r"\b\d{10,}\b", text):
        rules.append("Contains a long phone-number-like sequence")
    if re.search(r"[!]{2,}|\?{2,}|[A-Z]{6,}", message):
        rules.append("Contains unusually strong punctuation or capitalization")
    return rules

def get_spam_index() -> int:
    spam_index = 1
    id2label = getattr(model.config, "id2label", {}) or {}
    for idx, label in id2label.items():
        if "spam" in str(label).lower():
            spam_index = int(idx)
            break
    return spam_index

def model_predict(message: str) -> float:
    if not MODEL_LOADED:
        raise RuntimeError(
            "DistilBERT model is not available. Put the extracted "
            "'distilbert_sms_spam_model' folder in the project root."
        )
    inputs = tokenizer(
        message, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    spam_index = get_spam_index()
    return float(probs[spam_index])

def model_predict_batch(messages: List[str], batch_size: int = 32) -> List[float]:
    if not MODEL_LOADED:
        raise RuntimeError("DistilBERT model is not available.")
    
    if not messages:
        return []

    spam_index = get_spam_index()
    all_scores = []

    for i in range(0, len(messages), batch_size):
        chunk = messages[i:i + batch_size]
        chunk = [str(m) if str(m).strip() else " " for m in chunk]
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            for p in probs:
                all_scores.append(float(p[spam_index]))

    return all_scores

def format_verdict(message: str, spam_probability: float, threshold: float = 0.5):
    rules = rule_scan(message)
    final_score = spam_probability
    if rules and 0.35 <= spam_probability <= 0.65:
        final_score = min(1.0, spam_probability + min(0.15, 0.03 * len(rules)))

    label = "SPAM" if final_score >= threshold else "HAM"
    if final_score >= 0.85:
        risk = "HIGH"
    elif final_score >= 0.60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if label == "SPAM":
        if rules:
            explanation = (
                "The model classified this message as spam, and the rule-based "
                "agent detected: " + "; ".join(rules) + "."
            )
        else:
            explanation = (
                "The DistilBERT model classified this message as spam. "
                "No additional rule-based trigger was detected."
            )
    else:
        if rules:
            explanation = (
                "The model classified this message as ham. Some suspicious "
                "patterns were detected, but the model score remained below the threshold."
            )
        else:
            explanation = "The message appears to be a normal, non-spam message."

    return {
        "label": label,
        "spam_probability": round(final_score, 4),
        "spam_percentage": round(final_score * 100, 2),
        "risk": risk,
        "triggered_rules": rules,
        "explanation": explanation,
        "model_probability": round(spam_probability, 4),
        "device": str(device),
    }

def classify(message: str, threshold: float = 0.5):
    spam_probability = model_predict(message)
    return format_verdict(message, spam_probability, threshold)

def read_csv_safely(content: bytes) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig"]
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(BytesIO(content), encoding=enc)
        except Exception as exc:
            last_error = exc
            continue

    # Fallback with error replacement
    try:
        text = content.decode("utf-8", errors="replace")
        return pd.read_csv(StringIO(text))
    except Exception as exc:
        raise ValueError(f"Could not read CSV with supported encodings: {last_error or exc}")

def find_message_column(df: pd.DataFrame) -> str:
    common_names = [
        "message", "text", "sms", "v2", "sms_text", "msg", "content",
        "body", "input", "sentence", "text_message", "email_text", "email"
    ]
    # Check exact match first
    for target in common_names:
        for col in df.columns:
            if str(col).strip().lower() == target:
                return col

    # Check substring match
    for col in df.columns:
        c_low = str(col).strip().lower()
        if any(kw in c_low for kw in ["message", "text", "sms", "content", "body"]):
            return col

    # Pick string column with longest text on average
    string_cols = [c for c in df.columns if df[c].dtype == object]
    if string_cols:
        best_col = max(string_cols, key=lambda c: df[c].astype(str).str.len().mean())
        return best_col

    if len(df.columns) > 0:
        return df.columns[0]

    raise ValueError("No valid columns found in CSV.")

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "model_path": str(MODEL_DIR),
    }

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    try:
        return classify(request.message, request.threshold)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/api/batch")
async def batch(file: UploadFile = File(...)):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="DistilBERT model is not loaded.")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        content = await file.read()
        df = read_csv_safely(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}")

    try:
        message_col = find_message_column(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    raw_messages = df[message_col].fillna("").astype(str).tolist()
    # Filter out empty or whitespace-only messages if needed, or keep all rows
    scores = model_predict_batch(raw_messages, batch_size=32)

    results = []
    for msg, score in zip(raw_messages, scores):
        res = format_verdict(msg, score)
        results.append({
            "message": msg,
            "prediction": res["label"],
            "spam_percentage": res["spam_percentage"],
            "risk": res["risk"],
            "triggered_rules": " | ".join(res["triggered_rules"]),
            "explanation": res["explanation"],
        })

    return {
        "filename": file.filename,
        "detected_column": str(message_col),
        "rows": results,
        "total": len(results),
        "spam_count": sum(r["prediction"] == "SPAM" for r in results),
        "ham_count": sum(r["prediction"] == "HAM" for r in results),
    }

FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

