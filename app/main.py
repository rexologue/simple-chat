from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from openai import OpenAI


class Settings(BaseModel):
    vllm_base_url: str = Field(default="http://vllm-service:8000/v1", alias="VLLM_BASE_URL")
    vllm_model_name: str = Field(default="/app/model", alias="VLLM_MODEL_NAME")
    max_context_tokens: int = Field(default=8000, alias="MAX_CONTEXT_TOKENS")
    session_ttl_seconds: int = Field(default=600, alias="SESSION_TTL_SECONDS")

    class Config:
        populate_by_name = True
        frozen = True


settings = Settings(**{k: v for k, v in os.environ.items() if k in {
    "VLLM_BASE_URL",
    "VLLM_MODEL_NAME",
    "MAX_CONTEXT_TOKENS",
    "SESSION_TTL_SECONDS",
}})


tokenizer = AutoTokenizer.from_pretrained(settings.vllm_model_name)


class InitSessionRequest(BaseModel):
    vllm_api_key: str = Field(description="API key that will be used to access vLLM")


class InitSessionResponse(BaseModel):
    session_id: str
    expires_in: int


class SetSystemPromptRequest(BaseModel):
    session_id: str
    system_prompt: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    extra: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    reply: str
    finish_reason: Optional[str] = None
    total_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class StopSessionRequest(BaseModel):
    session_id: str


class ChatMessageInternal(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionData(BaseModel):
    system_prompt: str
    history: List[ChatMessageInternal]
    last_activity: datetime
    vllm_api_key: str


sessions: Dict[str, SessionData] = {}
sessions_lock = asyncio.Lock()


def default_system_prompt() -> str:
    return (
        "Ты полезный ассистент по ML и программированию. "
        "Отвечай на русском, будь максимально конкретным, "
        "приводи примеры кода и практические советы."
    )


def build_messages_for_model(session: SessionData, user_message: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": session.system_prompt}
    ]
    for message in session.history:
        messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": user_message})
    return messages


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return len(tokenizer.encode(text))


async def trim_history_to_fit(
    session: SessionData,
    new_user_message: str,
    max_context_tokens: int,
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    history = list(session.history)

    while True:
        candidate_session = SessionData(
            system_prompt=session.system_prompt,
            history=history,
            last_activity=session.last_activity,
            vllm_api_key=session.vllm_api_key,
        )
        messages = build_messages_for_model(candidate_session, new_user_message)
        input_tokens = estimate_tokens(messages)

        if input_tokens + max_new_tokens <= max_context_tokens:
            return messages

        if not history:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Запрос слишком длинный: контекст модели заполнен даже без истории.",
            )

        history.pop(0)


async def get_session_or_404(session_id: str) -> SessionData:
    async with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена.")
        session.last_activity = datetime.utcnow()
        sessions[session_id] = session
        return session


def ensure_valid_vllm_api_key(api_key: str) -> None:
    client = OpenAI(api_key=api_key, base_url=settings.vllm_base_url)

    try:
        client.models.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid VLLM_API_KEY: vLLM denied access",
        ) from exc


app = FastAPI(title="Qwen Session Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/init_session", response_model=InitSessionResponse)
async def init_session(request: InitSessionRequest):
    ensure_valid_vllm_api_key(request.vllm_api_key)

    session_id = str(uuid.uuid4())
    session = SessionData(
        system_prompt=default_system_prompt(),
        history=[],
        last_activity=datetime.utcnow(),
        vllm_api_key=request.vllm_api_key,
    )

    async with sessions_lock:
        sessions[session_id] = session

    return InitSessionResponse(session_id=session_id, expires_in=settings.session_ttl_seconds)


@app.post("/set_system_prompt")
async def set_system_prompt(request: SetSystemPromptRequest):
    async with sessions_lock:
        session = sessions.get(request.session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена.")
        session.system_prompt = request.system_prompt
        session.last_activity = datetime.utcnow()
        sessions[request.session_id] = session
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = await get_session_or_404(request.session_id)

    llm_client = OpenAI(api_key=session.vllm_api_key, base_url=settings.vllm_base_url)

    messages = await trim_history_to_fit(
        session=session,
        new_user_message=request.message,
        max_context_tokens=settings.max_context_tokens,
        max_new_tokens=request.max_tokens,
    )

    input_tokens = estimate_tokens(messages)
    extra_body: Dict[str, Any] = dict(request.extra or {})

    try:
        completion = llm_client.chat.completions.create(
            model=settings.vllm_model_name,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            extra_body=extra_body or None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM error: {exc}") from exc

    choice = completion.choices[0]
    reply_text = choice.message.content or ""

    async with sessions_lock:
        current_session = sessions.get(request.session_id)
        if current_session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Сессия не найдена.")
        current_session.history.append(ChatMessageInternal(role="user", content=request.message))
        current_session.history.append(ChatMessageInternal(role="assistant", content=reply_text))
        current_session.last_activity = datetime.utcnow()
        sessions[request.session_id] = current_session

    usage = completion.usage
    total_tokens = getattr(usage, "total_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)

    return ChatResponse(
        reply=reply_text,
        finish_reason=choice.finish_reason,
        total_tokens=total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@app.post("/stop_session")
async def stop_session(request: StopSessionRequest):
    async with sessions_lock:
        sessions.pop(request.session_id, None)
    return {"status": "stopped"}


async def cleanup_sessions_task():
    while True:
        await asyncio.sleep(60)
        now = datetime.utcnow()
        expired: list[str] = []
        async with sessions_lock:
            for session_id, session in list(sessions.items()):
                if now - session.last_activity > timedelta(seconds=settings.session_ttl_seconds):
                    expired.append(session_id)
            for session_id in expired:
                sessions.pop(session_id, None)


@app.on_event("startup")
async def on_startup() -> None:
    asyncio.create_task(cleanup_sessions_task())


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}
