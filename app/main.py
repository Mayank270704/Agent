from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services.chat import ChatService

app = FastAPI(title=settings.app_name)
chat_service = ChatService()


class ChatRequest(BaseModel):
    message: str
    # Step 13: optional conversation-scope key, NOT identity/auth (see
    # app/agent/memory.py). Omitted -> ChatService's single legacy
    # conversation, exactly like before Step 13.
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": settings.app_name}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        # An explicit-but-blank session_id (e.g. "" or "   ") is rejected by
        # InMemorySessionMemoryStore itself as a ValueError -> the existing
        # 400 handler below already covers it; no new validation is added
        # here (see the Step 13 report on avoiding duplicated validation).
        reply = chat_service.ask(request.message, session_id=request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=True)
