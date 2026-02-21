from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import requests
import os

app = FastAPI()

HF_API_URL = "https://huggingface.co/alsgyu/sentiment-analysis-fine-tuned-model"
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API running with HuggingFace Inference API"}


@app.post("/predict")
def predict(request: TextRequest):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN not set in environment"}

    payload = {"inputs": request.text}

    try:
        response = requests.post(
            HF_API_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

    return {
        "status_code": response.status_code,
        "raw_text": response.text  # 🔥 JSON 파싱 안함
    }


@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """
    <html>
        <body>
            <h2>Sentiment Analysis Test</h2>
            <form id="form">
                <input type="text" id="text" />
                <button type="submit">Predict</button>
            </form>
            <pre id="result"></pre>

            <script>
            document.getElementById('form').onsubmit = async (e) => {
                e.preventDefault();
                const text = document.getElementById('text').value;
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text})
                });
                const data = await res.json();
                document.getElementById('result').innerText = JSON.stringify(data, null, 2);
            }
            </script>
        </body>
    </html>
    """