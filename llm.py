"""
LLM модуль с поддержкой GigaChat OAuth2
Автоматически получает и обновляет токен доступа
"""
import os
import httpx
import uuid
import base64
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class GigaChatLLM:
    """LLM клиент с автоматическим управлением OAuth2 токеном для GigaChat"""
    
    def __init__(
        self,
        client_secret: str,
        base_url: str = "https://api.giga.chat/v1",
        model: str = "GigaChat",
        scope: str = "GIGACHAT_API_PERS"
    ):
        """
        Args:
            client_secret: Client Secret из Sber Developer Studio
            base_url: Базовый URL для GigaChat API
            model: Название модели (GigaChat, GigaChat-Pro, GigaChat-Max)
            scope: Scope для физ/юр лиц (GIGACHAT_API_PERS или GIGACHAT_API_CORP)
        """
        self.client_secret = client_secret
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.scope = scope
        
        # OAuth2 параметры
        self.oauth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        # HTTP клиент с SSL verify=False для корп. сертификата Сбера
        self.http_client = httpx.Client(verify=False, timeout=60.0)
    
    def _get_authorization_header(self) -> str:
        """Создаёт Authorization заголовок для OAuth2 запроса"""
        # Client Secret уже содержит Client ID и Secret в формате UUID
        # Кодируем в Base64
        credentials = base64.b64encode(self.client_secret.encode()).decode()
        return f"Basic {credentials}"
    
    def _refresh_token(self) -> None:
        """Получает новый access token через OAuth2"""
        try:
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": self._get_authorization_header()
            }
            
            data = {"scope": self.scope}
            
            logger.info("Запрашиваем новый access token от GigaChat...")
            response = self.http_client.post(
                self.oauth_url,
                headers=headers,
                data=data
            )
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            
            # Токен действует expires_at миллисекунд
            expires_in_ms = token_data.get("expires_at", 1800000)  # По умолчанию 30 минут
            expires_in_seconds = expires_in_ms / 1000
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in_seconds - 60)
            
            logger.info(f"Access token получен, истекает через {expires_in_seconds/60:.1f} минут")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"OAuth2 ошибка {e.response.status_code}: {e.response.text}")
            raise Exception(f"Не удалось получить access token: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Ошибка при получении токена: {e}")
            raise
    
    def _ensure_token(self) -> None:
        """Проверяет валидность токена и обновляет при необходимости"""
        if self.access_token is None or self.token_expires_at is None:
            self._refresh_token()
        elif datetime.now() >= self.token_expires_at:
            logger.info("Access token истёк, обновляем...")
            self._refresh_token()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Отправляет запрос к GigaChat API
        
        Args:
            messages: Список сообщений в формате OpenAI
            temperature: Температура генерации
            max_tokens: Максимум токенов в ответе
            stream: Потоковая генерация (пока не поддерживается)
        
        Returns:
            Ответ в формате OpenAI-compatible
        """
        self._ensure_token()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            url = f"{self.base_url}/chat/completions"
            logger.info(f"Отправляем запрос к GigaChat: {self.model}")
            
            response = self.http_client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_text = e.response.text
            
            logger.error(f"GigaChat HTTP {status}: {error_text}")
            
            # Пробуем распарсить ошибку
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", error_text)
            except:
                error_msg = error_text
            
            raise Exception(f"GigaChat API ошибка {status}: {error_msg}")
        
        except Exception as e:
            logger.error(f"Ошибка при обращении к GigaChat: {e}")
            raise
    
    def close(self):
        """Закрывает HTTP клиент"""
        self.http_client.close()


class LLMClient:
    """
    Универсальный LLM клиент с автоопределением провайдера
    Совместим со старым интерфейсом бота
    """
    
    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.giga.chat/v1")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "GigaChat")
        
        # Определяем провайдера
        if "giga.chat" in self.base_url.lower() or "gigachat" in self.base_url.lower():
            self.provider = "gigachat"
            logger.info(f"Используем GigaChat провайдер: {self.model}")
            self.client = GigaChatLLM(
                client_secret=self.api_key,
                base_url=self.base_url,
                model=self.model
            )
        else:
            # Для других провайдеров используем OpenAI-совместимый клиент
            self.provider = "openai"
            logger.info(f"Используем OpenAI-совместимый провайдер")
            self.client = httpx.Client(timeout=60.0)
    
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Получает ответ от LLM (совместимый интерфейс)
        
        Returns:
            Текст ответа от модели
        """
        try:
            if self.provider == "gigachat":
                response = self.client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # OpenAI-совместимый запрос
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature
                }
                
                if max_tokens:
                    payload["max_tokens"] = max_tokens
                
                url = f"{self.base_url}/chat/completions"
                resp = self.client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                response = resp.json()
            
            # Извлекаем текст ответа
            return response["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Ошибка LLM запроса: {e}")
            raise
    
    def close(self):
        """Закрывает клиент"""
        if self.provider == "gigachat":
            self.client.close()
        else:
            self.client.close()


# Экспорт для совместимости
__all__ = ["LLMClient", "GigaChatLLM"]
