from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, model_validator


def _enforce_stop_tokens(text: str, stop: List[str]) -> str:
    for token in stop:
        if token and token in text:
            text = text.split(token, 1)[0]
    return text


class ChatYandexGPT(BaseChatModel):
    """LangChain совместимый клиент для YandexGPT."""

    model: str = Field(default="yandexgpt-lite")
    api_key: Optional[str] = Field(default=None)
    folder_id: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1800)
    base_url: str = Field(
        default="https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    )
    timeout: int = Field(default=60)

    @model_validator(mode="before")
    def _ensure_credentials(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        api_key = (
            values.get("api_key")
            or os.getenv("YANDEX_API_KEY")
            or os.getenv("YC_API_KEY")
        )
        folder_id = (
            values.get("folder_id")
            or os.getenv("YANDEX_FOLDER_ID")
            or os.getenv("YC_FOLDER_ID")
        )

        if not api_key:
            raise ValueError(
                "YANDEX_API_KEY (или YC_API_KEY) не найден. "
                "Укажи его в .env или передай параметром."
            )
        if not folder_id:
            raise ValueError(
                "YANDEX_FOLDER_ID (или YC_FOLDER_ID) не найден. "
                "Укажи его в .env или передай параметром."
            )

        values["api_key"] = api_key
        values["folder_id"] = folder_id
        return values

    @property
    def _llm_type(self) -> str:
        return "yandexgpt-chat"

    def _build_model_uri(self) -> str:
        return f"gpt://{self.folder_id}/{self.model}"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        converted: List[Dict[str, str]] = []
        for message in messages:
            role = role_map.get(message.type, "user")
            content = message.content
            if isinstance(content, list):
                # LangChain может прислать список блоков; склеиваем в текст.
                parts: List[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        parts.append(block.get("text") or block.get("json", ""))
                    else:
                        parts.append(str(block))
                text = "\n".join(part for part in parts if part)
            else:
                text = str(content)

            converted.append({"role": role, "text": text})
        return converted

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.base_url, headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "modelUri": self._build_model_uri(),
            "completionOptions": {
                "stream": False,
                "temperature": kwargs.get("temperature", self.temperature),
                "maxTokens": kwargs.get("max_tokens", self.max_tokens),
            },
            "messages": self._convert_messages(messages),
        }

        response_format = kwargs.get("response_format")
        if response_format:
            payload["responseFormat"] = response_format

        data = self._post(payload)

        try:
            alternative = data["result"]["alternatives"][0]
            text = alternative["message"]["text"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Неожиданный ответ YandexGPT: {data}") from exc

        if stop:
            text = _enforce_stop_tokens(text, stop)

        ai_message = AIMessage(content=text)
        generation = ChatGeneration(message=ai_message, text=text)

        llm_output = {
            "model": self.model,
            "usage": data.get("result", {}).get("usage"),
            "status": alternative.get("status"),
        }
        return ChatResult(generations=[generation], llm_output=llm_output)

