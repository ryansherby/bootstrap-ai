import os
import threading
import warnings

from openai import OpenAI

from model_definitions.base_model import BaseModel


class OpenAIModel(BaseModel):

    _api_key = None
    __client = None
    _lock = threading.Lock()

    def __init__(self, gpu_number=0, device="cuda"):
        super().__init__(gpu_number, device)

    @classmethod
    def _load_key_from_env(cls):
        key = os.getenv("OPENAI_API_KEY", "").strip()

        if key:
            return key

        path = os.getenv("OPENAI_API_KEY_PATH", "openai_api_key.txt")

        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except OSError as e:
                warnings.warn(
                    f"Couldn't read OPENAI_API_KEY_PATH={path}: {e}"
                )

        return None

    @classmethod
    def _client(cls):
        if cls.__client is None:
            with cls._lock:
                if cls.__client is None:
                    if not cls._api_key:
                        cls._api_key = cls._load_key_from_env()

                    if not cls._api_key:
                        raise ValueError(
                            "OpenAI API key not found. "
                            "Set OPENAI_API_KEY or OPENAI_API_KEY_PATH "
                            "environment variable."
                        )

                    cls.__client = OpenAI(api_key=cls._api_key, max_retries=0)

        return cls.__client

    @property
    def client(self):
        return self.__class__._client()