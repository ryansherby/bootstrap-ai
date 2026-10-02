import threading
import warnings
import os

from google import genai

from model_definitions.base_model import BaseModel


class GeminiModel(BaseModel):

    _api_key = None
    __client = None
    _lock = threading.Lock()

    def __init__(self, gpu_number=0, device="cuda"):
        super().__init__(gpu_number, device)

    @classmethod
    def _load_key_from_env(cls):
        key = os.getenv("GEMINI_API_KEY")

        if key:
            return key.strip()

        path = os.getenv(
            "GEMINI_API_KEY_PATH",
            "gemini_api_key.txt",
        )

        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except OSError as e:
                warnings.warn(
                    f"Couldn't read "
                    f"GEMINI_API_KEY_PATH={path}: {e}"
                )

        return None

    @classmethod
    def _client(cls):
        with cls._lock:
            if cls.__client is None:

                if not cls._api_key:
                    cls._api_key = cls._load_key_from_env()

                if not cls._api_key:
                    raise ValueError(
                        "Gemini API key not found. "
                        "Set GEMINI_API_KEY or "
                        "GEMINI_API_KEY_PATH."
                    )

                cls.__client = genai.Client(
                    api_key=cls._api_key,
                    http_options=genai.types.HttpOptions(
                        retry_options=genai.types.HttpRetryOptions(
                            attempts=1,
                        )
                    ),
                )

        return cls.__client

    @property
    def client(self):
        return self.__class__._client()