"""LLM клиент для работы с GigaChat через OAuth2"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GigaChatLLM:
    """Клиент для работы с GigaChat API через OAuth2"""
    
    def __init__(
        self,
        client_id: str,
        base_url: str = "https://api.giga.chat/v1",
        model: str = "GigaChat",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        self.client_id = client_id
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        self._access_token: str | None = None
        self._token_expires_at: float = 0
        
        self.client = httpx.AsyncClient(timeout=timeout, verify=False)
    
async def _get_access_token(self) -> str:
        """Получает OAuth2 токен от GigaChat"""
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        
        logger.info("Получаем новый OAuth2 токен от GigaChat...")
        
        try:
            auth_data = base64.b64encode(f"{self.client_id}:".encode()).decode()
            
            response = await self.client.post(
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": f"{time.time_ns()}",
                    "Authorization": f"Basic {auth_data}",
                },
                data="scope=GIGACHAT_API_PERS",
            )
            response.raise_for_status()
            
            data = response.json()
            self._access_token = data["access_token"]
            expires_in = data.get("expires_in", 1800)
            self._token_expires_at = time.time() + expires_in - 60
            
            logger.info("OAuth2 токен успешно получен")
            return self._access_token
            
        except Exception as e:
            logger.error(f"Ошибка получения OAuth2 токена: {e}")
            raise LLMError(f"Не удалось получить токен GigaChat: {e}")
    
    async def complete(self, messages: list[dict]) -> str:
        """Отправка запроса к GigaChat"""
        token = await self._get_access_token()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(f"Получен ответ от {self.model}, {len(content)} символов")
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка HTTP {e.response.status_code}: {e.response.text}")
            raise LLMError(f"GigaChat API вернул ошибку {e.response.status_code}")
        except Exception as e:
            logger.error(f"Ошибка LLM запроса: {e}")
            raise LLMError(f"Ошибка при запросе к GigaChat: {e}")
    
    async def close(self):
        """Закрывает HTTP клиент"""
        if self.client:
            await self.client.aclose()


class LLMClient:
    """Обёртка для работы с GigaChat (совместимость со старым кодом)"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        timeout: int = 120,
        proxy: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._client = GigaChatLLM(
            client_id=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
    
    async def complete(self, messages: list[dict]) -> str:
        """Отправка запроса к GigaChat"""
        return await self._client.complete(messages)
    
    async def close(self):
        """Закрывает клиент"""
        await self._client.close()


class LLMError(Exception):
    """Ошибка при запросе к LLM"""
    pass


FREE_PRESETS = {
    "gigachat": ("GigaChat", "https://api.giga.chat/v1", "GigaChat"),
    "gigachat-pro": ("GigaChat Pro", "https://api.giga.chat/v1", "GigaChat-Pro"),
}


__all__ = ["LLMClient", "GigaChatLLM", "LLMError", "FREE_PRESETS"]
