import logging
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import APIError, Groq
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().with_name(".env"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Groq(api_key=api_key)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str


def chat_bot(user_message: str):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_message}],
            model="openai/gpt-oss-120b",
        )
    except APIError as exc:
        logger.exception("Groq API request failed")
        raise HTTPException(
            status_code=502,
            detail=f"Groq API request failed (status {exc.status_code}). Check the server terminal for details.",
        ) from exc

    return chat_completion.choices[0].message.content


@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be empty.")
    return {"reply": chat_bot(message)}
    