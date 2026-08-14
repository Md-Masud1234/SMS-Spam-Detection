# SMS Spam Detection & Intelligent Message Filtering

Mini project using a fine-tuned DistilBERT classifier plus a simple rule-based explanation layer.

## Project structure

- `backend/main.py` - FastAPI REST API
- `frontend/` - standalone HTML/CSS/JavaScript UI
- `distilbert_sms_spam_model/` - **put your trained model here**
- `requirements.txt` - Python dependencies
- `test_messages.csv` - sample batch input

## 1. Add your trained model

From Google Colab, download and extract your trained model so this exists:

```text
distilbert_sms_spam_model/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

The model folder must be in the project root, next to `backend`.

## 2. Open VS Code terminal

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Start the backend

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Open:

http://127.0.0.1:8000/docs

Test `/api/health` first. It should say `model_loaded: true`.

## 4. Start the frontend

Install the VS Code **Live Server** extension.

Open `frontend/index.html`, right-click it and choose **Open with Live Server**.

The browser will open on a local address such as:

http://127.0.0.1:5500/frontend/index.html

## 5. Test

Try:

`Congratulations! You have won a free prize. Click here now!`

and:

`Hey, are you coming to college tomorrow?`

## Notes

- This package does not include the trained model because the model is created/downloaded from your Colab environment.
- The backend expects the conventional SMS dataset mapping `0=ham, 1=spam`, but it also checks the saved model's `id2label` metadata for a spam label.
- The rule layer is intended as an explanation/reasoning layer, not a replacement for the trained classifier.
