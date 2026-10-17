from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    index_path: Path = Field(default=Path("index.pkl"), validation_alias="INDEX_PATH")
    embeddings_path: Path | None = Field(
        default=None,
        validation_alias="EMBEDDINGS_PATH",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    port: int = Field(default=8000, validation_alias="PORT")
    page_size: int = Field(default=10, ge=1, le=50, validation_alias="PAGE_SIZE")


@lru_cache
def get_settings() -> Settings:
    return Settings()
