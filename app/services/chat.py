from __future__ import annotations

from app.agent.orchestrator import AgentOrchestrator
from app.config import settings
from app.models.llm import LLMClient


class ChatService:
    def __init__(self):
        self.llm = LLMClient(
            provider=settings.llm_provider,
            model_name=settings.model_name,
            api_key=settings.openai_api_key,
            base_url=settings.ollama_base_url,
        )
        self.orchestrator = AgentOrchestrator(llm_client=self.llm)
        self.conversation: list[dict[str, str]] = []

    def ask(self, user_message: str) -> str:
        if user_message is None or not user_message.strip():
            raise ValueError("User message cannot be empty.")

        cleaned_message = user_message.strip()
        self.conversation.append({"role": "user", "content": cleaned_message})

        result = self.orchestrator.process(cleaned_message)
        response = result.answer

        self.conversation.append({"role": "assistant", "content": response})
        return response
