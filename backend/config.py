from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VoiceGuard AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Forensic Thresholds
    SIMILARITY_THRESHOLD: float = 60.0
    
    class Config:
        env_file = ".env"

settings = Settings()
