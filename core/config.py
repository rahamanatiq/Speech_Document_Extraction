from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central, validated configuration — the only place env vars are read."""

    speech_provider: str = "mock"   # "mock" | "whisper"
    ocr_provider: str = "mock"      # "mock" | "easyocr"
    whisper_model_size: str = "medium"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()