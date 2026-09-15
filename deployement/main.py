import os
import secrets

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ValidationError
from tokenizers import Tokenizer

# ---- Настройки (берутся из переменных окружения) ----
VLLM_URL = os.getenv("VLLM_URL", "http://llm:8000/v1/chat/completions")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "/adapters/3b-adapter/tokenizer.json")

MODEL_NAME = "phishing"      # имя из --lora-modules
MAX_CHARS = 200_000          # защита от огромных запросов
MAX_EMAIL_TOKENS = 1400      # 2048 контекст - 600 на ответ - системный промпт
MAX_NEW_TOKENS = 600

if not SERVICE_API_KEY:
    raise RuntimeError("SERVICE_API_KEY не задан")

# Промпт ДОЛЖЕН совпадать с тем, что был при обучении и оценке
SYSTEM_PROMPT = (
    "You are an advanced AI security analyst specialized in email threat detection. "
    "Analyze the provided email and respond with a JSON object containing: "
    "is_phishing, confidence_score, threat_type, risk_level, indicators, "
    "mitigation_recommendations, analysis_summary."
)


# ---- Схема ответа (как в README) ----
class Indicator(BaseModel):
    category: str
    finding: str
    severity: str
    explanation: str


class Mitigation(BaseModel):
    immediate_actions: list[str]
    preventive_measures: list[str]
    reporting_guidance: str


class Analysis(BaseModel):
    is_phishing: bool
    confidence_score: float
    threat_type: str
    risk_level: str
    indicators: list[Indicator]
    mitigation_recommendations: Mitigation
    analysis_summary: str


class EmailIn(BaseModel):
    email_text: str


tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
app = FastAPI(title="Phishing analyzer (demo)")


def cut_email(text):
    """Обрезает письмо по токенам. Возвращает (текст, был_ли_обрезан)."""
    ids = tokenizer.encode(text).ids
    if len(ids) <= MAX_EMAIL_TOKENS:
        return text, False
    return tokenizer.decode(ids[:MAX_EMAIL_TOKENS]), True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(item: EmailIn, x_api_key: str = Header(default="")):
    # 1. Проверка ключа
    if not secrets.compare_digest(x_api_key, SERVICE_API_KEY):
        raise HTTPException(status_code=401, detail="bad api key")

    # 2. Проверка размера
    if len(item.email_text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail="email too large")

    email_text, truncated = cut_email(item.email_text)

    # 3. Запрос к vLLM
    body = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": email_text},
        ],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "phishing_analysis",
                "schema": Analysis.model_json_schema(),
            },
        },
    }
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(VLLM_URL, json=body, headers=headers)
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="model server unavailable")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="model server error")

    raw = r.json()["choices"][0]["message"]["content"]

    # 4. Проверка ответа модели
    try:
        result = Analysis.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(status_code=502, detail="model returned invalid JSON")

    # Текст письма НЕ логируем и не сохраняем
    return {"truncated": truncated, "result": result.model_dump()}
