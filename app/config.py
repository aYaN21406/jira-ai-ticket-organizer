from pydantic import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Jira configuration
    jira_base_url: str
    jira_email: str
    jira_api_token: str
    jira_project_key: str

    # Service configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Embeddings / NLP
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Database (if using Postgres + pgvector later)
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "jira_ai"
    db_password: str = "jira_ai_password"
    db_name: str = "jira_ai"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
