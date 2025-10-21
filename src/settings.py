from pydantic import BaseSettings

class Settings(BaseSettings):
    # We rely on env var and/or Streamlit's secrets at runtime
    openai_api_key: str | None = None

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()