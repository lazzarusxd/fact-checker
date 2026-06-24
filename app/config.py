from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""

    groq_base_url: str = "https://api.groq.com/openai/v1"

    llm_model: str = "llama-3.3-70b-versatile"

    sqlite_path: str = "fact_checker.db"

    max_debate_rounds: int = 2

    rag_top_k: int = 3

    web_results: int = 4

    max_sub_claims: int = 4

    embedding_model: str = "all-MiniLM-L6-v2"

    corpus_dir: str = "data/corpus"

    learned_top_k: int = 3

    learned_ttl_days: int = 30

    host: str = "127.0.0.1"

    port: int = 8000

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
